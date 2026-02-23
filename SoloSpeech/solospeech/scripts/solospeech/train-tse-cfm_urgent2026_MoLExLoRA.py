import yaml
import random
import argparse
import os
import time
import re
from typing import Iterable, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
import pandas as pd
import csv
from tqdm import tqdm
from torch.utils.data import DataLoader

from accelerate import Accelerator
from diffusers import FlowMatchEulerDiscreteScheduler

# ===== 你的项目依赖 =====
from model.solospeech.conditioners import SoloSpeech_TSR
from dataset import TSEDataset2

# ---------------------------
# ==== MoLEx Linear & FFN ===
# ---------------------------

class MoLExLinear(nn.Module):
    """
    Linear + Mixture of LoRA Experts (MoLEx).

    y = x W^T + scaling * sum_{i in TopK(x)} g_i(x) * (B_i A_i x)

    - W: 基础 weight（可加载预训练、可冻结）
    - (A_i, B_i): 第 i 个 LoRA expert
    - g_i(x): 带噪声的 top-k gating 权重
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        r: int = 8,
        num_experts: int = 4,
        k: int = 2,
        alpha: float = 16.0,
        noise_std: float = 1.0,
        bias: bool = True,
    ):
        super().__init__()
        assert 1 <= k <= num_experts

        self.in_features = in_features
        self.out_features = out_features
        self.r = r
        self.num_experts = num_experts
        self.k = k
        self.alpha = alpha
        self.scaling = alpha / r
        self.noise_std = noise_std

        # base linear
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias = nn.Parameter(torch.zeros(out_features)) if bias else None

        # LoRA experts: A: [E, r, in], B: [E, out, r]
        self.lora_A = nn.Parameter(torch.zeros(num_experts, r, in_features))
        self.lora_B = nn.Parameter(torch.zeros(num_experts, out_features, r))

        # gating network: x -> logits over experts
        self.gate = nn.Linear(in_features, num_experts, bias=True)

        self.reset_parameters()

    def reset_parameters(self):
        # base linear
        nn.init.kaiming_uniform_(self.weight, a=5 ** 0.5)
        if self.bias is not None:
            bound = 1 / (self.in_features ** 0.5)
            nn.init.uniform_(self.bias, -bound, bound)

        # ===== 关键修改：只把 B 置零，A 用随机初始化 =====
        nn.init.kaiming_uniform_(self.lora_A, a=5 ** 0.5)  # A 随机
        nn.init.zeros_(self.lora_B)                       # B = 0

        # gating network：保持 0 也没问题（初始时接近均匀 + 噪声）
        nn.init.zeros_(self.gate.weight)
        if self.gate.bias is not None:
            nn.init.zeros_(self.gate.bias)


    def freeze_base(self):
        """LoRA finetune 时调用：冻结 base W 和 bias。"""
        self.weight.requires_grad_(False)
        if self.bias is not None:
            self.bias.requires_grad_(False)

    def forward(self, x):
        """
        x: [B, T, D] 或 [B, D] 或 [*, D]
        """
        orig_shape = x.shape
        last_dim = orig_shape[-1]
        assert last_dim == self.in_features

        x_flat = x.reshape(-1, last_dim)  # [N, D]

        # 1) base linear
        base_out = F.linear(x_flat, self.weight, self.bias)  # [N, out]

        # 2) noisy top-k gating
        logits = self.gate(x_flat)  # [N, E]
        if self.noise_std > 0.0 and self.training:
            noise = torch.randn_like(logits) * self.noise_std
            logits = logits + noise

        topk_logits, topk_indices = torch.topk(logits, self.k, dim=-1)   # [N, k]
        topk_gates = F.softmax(topk_logits, dim=-1)                       # [N, k]

        # 3) LoRA experts
        # xA: [N, E, r]
        xA = torch.einsum("nd,erd->ner", x_flat, self.lora_A)
        # delta_all: [N, E, out]
        delta_all = torch.einsum("ner,eor->neo", xA, self.lora_B)

        # 4) gather top-k experts per token
        N = x_flat.shape[0]
        _, E, O = delta_all.shape
        idx = topk_indices.unsqueeze(-1).expand(N, self.k, O)  # [N, k, out]
        delta_topk = torch.gather(delta_all, 1, idx)           # [N, k, out]

        gates = topk_gates.unsqueeze(-1)                       # [N, k, 1]
        lora_update = (delta_topk * gates).sum(dim=1)          # [N, out]

        out_flat = base_out + self.scaling * lora_update
        out = out_flat.view(*orig_shape[:-1], self.out_features)
        return out


class MoLExFeedForward(nn.Module):
    """
    用 MoLExLinear 替换原始 FFN 里的 w1, w2.
    结构与原版 FeedForward 保持一致：
        w1: in -> 2*hidden (分成 x / gate)
        w2: hidden -> in

    *注意：这个类目前没有直接在 SoloSpeech 中用，
    我们是通过 replace 把已有 FeedForward 里的 w1/w2 换成 MoLExLinear。
    如果你想直接替换整个 FFN，可以用这个类。
    """

    def __init__(
        self,
        in_features: int,
        hidden_size: int,
        multiple_of: int,
        ffn_dim_multiplier=None,
        bias: bool = True,
        # MoLEx 超参：
        lora_rank: int = 8,
        num_experts: int = 4,
        top_k: int = 2,
        lora_alpha: float = 16.0,
        gate_noise_std: float = 1.0,
    ):
        super().__init__()
        hidden_size = int(2 * hidden_size / 3)
        if ffn_dim_multiplier is not None:
            hidden_size = int(ffn_dim_multiplier * hidden_size)
        hidden_size = multiple_of * ((hidden_size + multiple_of - 1) // multiple_of)

        self.hidden_dim = hidden_size

        self.w1 = MoLExLinear(
            in_features=in_features,
            out_features=2 * hidden_size,
            r=lora_rank,
            num_experts=num_experts,
            k=top_k,
            alpha=lora_alpha,
            noise_std=gate_noise_std,
            bias=bias,
        )

        self.w2 = MoLExLinear(
            in_features=hidden_size,
            out_features=in_features,
            r=lora_rank,
            num_experts=num_experts,
            k=top_k,
            alpha=lora_alpha,
            noise_std=gate_noise_std,
            bias=bias,
        )

    def freeze_base(self):
        self.w1.freeze_base()
        self.w2.freeze_base()

    def forward(self, x):
        x_proj, gate = self.w1(x).chunk(2, dim=-1)
        x_act = F.silu(x_proj) * gate
        out = self.w2(x_act)
        return out


def count_parameters(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable







# ---------------------------
# ===== MoLEx LoRA utils ===
# ---------------------------

def _replace_linear_with_molex(
    parent: nn.Module,
    name: str,
    r: int,
    num_experts: int,
    k: int,
    alpha: float,
    noise_std: float,
):
    """
    用 MoLExLinear 替换 parent.name 这个 nn.Linear。
    会把原来的 weight/bias 拷贝到 MoLEx 的 base 里。
    """
    old: nn.Linear = getattr(parent, name)
    assert isinstance(old, nn.Linear), f"{name} is not nn.Linear"

    new = MoLExLinear(
        in_features=old.in_features,
        out_features=old.out_features,
        r=r,
        num_experts=num_experts,
        k=k,
        alpha=alpha,
        noise_std=noise_std,
        bias=(old.bias is not None),
    )

    # 把预训练线性层权重拷贝过去，作为 MoLEx 的 base W/b
    with torch.no_grad():
        new.weight.copy_(old.weight.data)
        if old.bias is not None and new.bias is not None:
            new.bias.copy_(old.bias.data)

    setattr(parent, name, new)


def apply_molex(
    model: nn.Module,
    r: int = 8,
    alpha: float = 16.0,
    num_experts: int = 4,
    k: int = 2,
    noise_std: float = 1.0,
    target_keywords: Iterable[str] = ("mlp.w1", "mlp.w2"),
    use_regex: bool = False,
) -> Tuple[int, int]:
    """
    遍历模型，把匹配的 nn.Linear 替换成 MoLExLinear。

    默认只匹配 DiTBlock.mlp 里的 w1 / w2（名字里包含 'mlp.w1' 或 'mlp.w2'）。
    你也可以通过 --molex-targets 自定义匹配规则。

    返回 (hit, replaced) 统计。
    """
    hit, replaced = 0, 0

    def name_matches(full_name: str) -> bool:
        if use_regex:
            return any(re.search(p, full_name) for p in target_keywords)
        return any(k in full_name for k in target_keywords)

    for full_name, module in list(model.named_modules()):
        for child_name, child in list(module.named_children()):
            if isinstance(child, nn.Linear) and name_matches(f"{full_name}.{child_name}"):
                print("[DEBUG] replacing", f"{full_name}.{child_name}")
                hit += 1
                _replace_linear_with_molex(
                    parent=module,
                    name=child_name,
                    r=r,
                    num_experts=num_experts,
                    k=k,
                    alpha=alpha,
                    noise_std=noise_std,
                )
                replaced += 1

    return hit, replaced


def only_optimize_molex(model: nn.Module):
    """
    冻结除 MoLEx LoRA 以外的所有参数：
    - Base W/b 冻结
    - 其他模块全部 requires_grad=False
    - 只训练 lora_A, lora_B, gate
    """
    # 先全局冻结
    for p in model.parameters():
        p.requires_grad_(False)

    for m in model.modules():
        if isinstance(m, MoLExLinear):
            # 冻结 base 权重 / bias
            m.freeze_base()

            # 开启 LoRA experts & gating 网络
            m.lora_A.requires_grad_(True)
            m.lora_B.requires_grad_(True)
            for p in m.gate.parameters():
                p.requires_grad_(True)


def safe_load_optimizer(optimizer: torch.optim.Optimizer, ckpt_opt_state: dict,
                        keep_hparams: bool = True, lr: float = None):
    """
    仅加载 state（动量等），保留当前 param_groups（从而保留“只训练 MoLExLoRA”和你设置的超参）。
    """
    cur = optimizer.state_dict()
    new_state = {
        "state": ckpt_opt_state.get("state", {}),
        "param_groups": cur["param_groups"] if keep_hparams else ckpt_opt_state.get("param_groups", cur["param_groups"])
    }
    if keep_hparams and lr is not None:
        for g in new_state["param_groups"]:
            g["lr"] = float(lr)
    optimizer.load_state_dict(new_state)

# ---------------------------
# ===== /MoLEx LoRA utils ===
# ---------------------------


def build_parser():
    parser = argparse.ArgumentParser()

    # 数据
    parser.add_argument('--train-clean', type=str, default='/export/fs05/tcao7/urgent2026/simulation_train_resample/clean/')
    parser.add_argument('--train-reverb', type=str, default='/export/fs05/tcao7/urgent2026/simulation_train_resample/noisy/')
    parser.add_argument('--sample-rate', type=int, default=16000)
    parser.add_argument('--vae-rate', type=int, default=50)
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--min-length', type=float, default=3.0)
    parser.add_argument("--num-infer-steps", type=int, default=50)

    # 训练
    parser.add_argument("--amp", type=str, default='fp16')
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--num-workers', type=int, default=16)
    parser.add_argument('--num-threads', type=int, default=1)
    parser.add_argument('--save-every', type=int, default=1)
    parser.add_argument("--adam-epsilon", type=float, default=1e-08)

    # 模型
    parser.add_argument('--diffusion-config', type=str, default='./config/base_solospeech_tse_cfm_urgent2026_MoLExLoRA.yaml')
    parser.add_argument('--autoencoder-path', type=str, default='./pretrained_models/audio-vae.pt')

    # 启动方式
    parser.add_argument('--resume-from', type=str, default=None,
                        help='Resume training from a training checkpoint (model + optimizer + progress if available)')
    parser.add_argument('--pretrained', type=str,
                        default="/export/fs05/tcao7/enhance/SoloSpeech/solospeech/base_solospeech_tse_cfm_urgent2026_baseline_full_ckpt/pretrained.pt",
                        help='Initialize from a pretrained base model (model-only), then start a fresh MoLEx-LoRA finetune')

    # 优化
    parser.add_argument('--learning-rate', type=float, default=1e-4)
    parser.add_argument('--beta1', type=float, default=0.9)
    parser.add_argument('--beta2', type=float, default=0.999)
    parser.add_argument('--weight-decay', type=float, default=1e-4)

    # 日志
    parser.add_argument('--random-seed', type=int, default=2025)
    parser.add_argument('--log-step', type=int, default=50)
    parser.add_argument('--log-dir', type=str, default='logs/')
    parser.add_argument('--save-dir', type=str, default='ckpt/')

    # ===== MoLEx-LoRA =====
    parser.add_argument('--use-molex', action='store_true',
                        help='Enable MoLEx-LoRA fine-tuning (Mixture of LoRA Experts)')
    parser.add_argument('--molex-rank', type=int, default=8)
    parser.add_argument('--molex-alpha', type=float, default=16.0)
    parser.add_argument('--molex-num-experts', type=int, default=5)
    parser.add_argument('--molex-top-k', type=int, default=3)
    parser.add_argument('--molex-noise-std', type=float, default=1.0)
    parser.add_argument('--molex-targets', type=str,
                        default='mlp.w1,mlp.w2',
                        help='Comma-separated substrings or regex to match Linear layers for MoLEx')
    parser.add_argument('--molex-targets-regex', action='store_true',
                        help='Interpret --molex-targets as regex patterns')

    return parser


def masked_mse_loss(predictions, targets, mask=None):
    if mask is not None:
        mask = mask.unsqueeze(-1).long()
        mse = (predictions - targets) ** 2
        masked_mse = mse * mask
        loss = masked_mse.sum() / mask.sum()
    else:
        mse = (predictions - targets) ** 2
        loss = mse.mean()
    return loss


def load_candidates(meta_tsv):
    """Load rows where duration<20s and augmentation!='none'."""
    candidates = []
    with open(meta_tsv, newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for r in reader:
            try:
                fs = float(r["fs"])
                length = float(r["length"])
                dur = length / fs if fs > 0 else float("inf")
            except Exception:
                continue

            aug = (r.get("augmentation") or "").strip().lower()
            if dur < 20.0 and aug != "none":
                candidates.append({
                    "id": r.get("id"),
                    "duration": dur,
                    "noisy_path": r.get("noisy_path"),
                    "clean_path": r.get("clean_path"),
                    "augmentation": r.get("augmentation"),
                })
    # ensure uniqueness by id (in case of duplicates)
    uniq = {}
    for c in candidates:
        if c["id"] not in uniq:
            uniq[c["id"]] = c
    return list(uniq.values())


def random_pack(candidates, target_seconds, tol_seconds, seed=2026):
    """
    Randomly shuffle then greedily accumulate until within tolerance or just above target.
    Returns the selected list and the total duration.
    """
    rng = random.Random(seed)
    pool = candidates[:]  # copy
    rng.shuffle(pool)

    selected = []
    total = 0.0
    for c in pool:
        # never add same id twice; pool is unique so this check is just defensive
        if any(c["id"] == s["id"] for s in selected):
            continue
        # add and check
        selected.append(c)
        total += c["duration"]
        if total >= target_seconds - tol_seconds:
            # we're at or near the target; if also not overshooting too much, stop
            if abs(total - target_seconds) <= tol_seconds or total >= target_seconds:
                break

    # If still below target (not enough data), return everything we have.
    return selected, total


def main():
    parser = build_parser()
    args = parser.parse_args()

    with open(args.diffusion_config, 'r') as fp:
        args.diff_config = yaml.safe_load(fp)

    args.v_prediction = args.diff_config["ddim"]["v_prediction"]
    args.log_dir = args.log_dir.replace('log', "30hour/" + args.diff_config["system"] + '_log')
    args.save_dir = args.save_dir.replace('ckpt', "30hour/" + args.diff_config["system"] + '_ckpt')

    os.makedirs(os.path.join(args.log_dir, 'audio/gt'), exist_ok=True)
    os.makedirs(args.save_dir, exist_ok=True)

    # 固定随机性
    random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)

    torch.set_num_threads(args.num_threads)
    if torch.cuda.is_available():
        args.device = 'cuda'
        torch.cuda.manual_seed(args.random_seed)
        torch.cuda.manual_seed_all(args.random_seed)
        torch.backends.cuda.matmul.allow_tf32 = True
        if torch.backends.cudnn.is_available():
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    else:
        args.device = 'cpu'

    META_TSV = "/export/fs05/tcao7/urgent2026/simulation_train/log/meta.tsv"
    TARGET_HOURS = 30.0             # select 12-hour data to finetune
    TOLERANCE_SECONDS = 300         # how close to target to aim for (±5 min)
    SEED = 2026                     # change for a different random selection

    target_seconds = TARGET_HOURS * 3600.0
    candidates = load_candidates(META_TSV)

    if not candidates:
        raise SystemExit("No candidates found with duration<20s and augmentation!='none'.")

    selected, total = random_pack(candidates, target_seconds, TOLERANCE_SECONDS, SEED)

    noisy_paths = sorted([s["noisy_path"] for s in selected if s["noisy_path"]])
    noisy_paths = [item.replace("simulation_train", "simulation_train_resample") for item in noisy_paths]
    clean_paths = sorted([s["clean_path"] for s in selected if s["clean_path"]])
    clean_paths = [item.replace("simulation_train", "simulation_train_resample") for item in clean_paths]

    print(f"kept {len(noisy_paths)} noisy files and {len(clean_paths)} clean files")

    train_set = TSEDataset2(reverb_dir=noisy_paths, clean_dir=clean_paths, debug=args.debug)
    train_loader = DataLoader(
        train_set,
        num_workers=args.num_workers,
        batch_size=args.batch_size,
        shuffle=True,
        pin_memory=True,
        collate_fn=train_set.collate,
    )

    accelerator = Accelerator(mixed_precision=args.amp)

    # ===== 1) 构建模型 =====
    model = SoloSpeech_TSR(args.diff_config['diffwrap']['UDiT'])
    total_params = sum(p.nelement() for p in model.parameters())
    print("Number of parameters: %.2fM" % (total_params / 1e6))

    # ===== 2) 权重来源判定 =====
    resume_path = args.resume_from if (args.resume_from and os.path.exists(args.resume_from)) else None
    pretrained_path = args.pretrained if (args.pretrained and os.path.exists(args.pretrained)) else None

    if resume_path:
        print(f"[Startup] resume-from = {resume_path} (training checkpoint)")
    elif pretrained_path:
        print(f"[Startup] pretrained = {pretrained_path} (base weights only)")
        ckpt = torch.load(pretrained_path, map_location='cpu')
        # 兼容 {"model": state_dict} / 直接 state_dict
        state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
        _ = model.load_state_dict(state, strict=False)
        print(f"Loaded pretrained base weights from {pretrained_path}")
    else:
        print("[Startup] training from scratch (no resume, no pretrained)")

    # ===== 3) 注入 MoLEx-LoRA（在 prepare 之前） =====
    if args.use_molex:
        molex_targets = tuple([s.strip() for s in args.molex_targets.split(",") if s.strip()])
        hit, replaced = apply_molex(
            model,
            r=args.molex_rank,
            alpha=args.molex_alpha,
            num_experts=args.molex_num_experts,
            k=args.molex_top_k,
            noise_std=args.molex_noise_std,
            target_keywords=molex_targets,
            use_regex=args.molex_targets_regex,
        )
        print(f"[MoLEx] matched {hit} Linear layers, replaced {replaced}")
        only_optimize_molex(model)  # 冻结非 MoLEx 参数

        # --- DEBUG 1: 看看 MoLEx 层有多少个 ---
        molex_param_count = 0
        total_trainable = 0
        for name, p in model.named_parameters():
            if p.requires_grad:
                total_trainable += p.numel()
                if "lora_A" in name or "lora_B" in name or "gate" in name:
                    molex_param_count += p.numel()
        print(f"[DEBUG] trainable params = {total_trainable}, MoLEx params = {molex_param_count}")
        assert total_trainable > 0, "No trainable params! Check use_molex / apply_molex."

    # ===== 4) 构建优化器（只含 requires_grad 参数） =====
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.learning_rate,
        betas=(args.beta1, args.beta2),
        weight_decay=args.weight_decay,
        eps=args.adam_epsilon
    )

    # ===== 5) 若是 resume，加载 checkpoint 的模型与优化器状态、进度 =====
    global_step = 0
    start_epoch = 0
    if resume_path:
        ckpt = torch.load(resume_path, map_location='cpu')
        missing, unexpected = model.load_state_dict(ckpt["model"], strict=False)
        if missing:
            print(f"[Resume] Missing keys (often due to new MoLEx params): {len(missing)}")
        if unexpected:
            print(f"[Resume] Unexpected keys in checkpoint: {len(unexpected)}")

        # 恢复优化器动量，但保留当前 param_groups 与超参（MoLEx-only）
        if "optimizer" in ckpt:
            try:
                safe_load_optimizer(optimizer, ckpt["optimizer"], keep_hparams=True, lr=args.learning_rate)
                print("[Resume] optimizer state loaded (kept current param_groups & hparams)")
            except Exception as e:
                print(f"[Resume] skip loading optimizer: {e}")

        global_step = ckpt.get("global_step", 0)
        start_epoch = ckpt.get("epoch", -1) + 1
        print(f"Resuming from {resume_path}, start epoch {start_epoch}, global_step {global_step}")

    # ===== 6) 准备调度器 =====
    noise_scheduler = FlowMatchEulerDiscreteScheduler(**args.diff_config["ddim"]['diffusers'])

    # ===== 7) 放设备 & prepare =====
    model.to(accelerator.device)
    model, optimizer, train_loader = accelerator.prepare(model, optimizer, train_loader)

    # ===== 8) 训练循环 =====
    def masked_mse(pred, target, mask):
        return masked_mse_loss(pred, target, mask)

    losses = 0.0
    total, trainable = count_parameters(model)
    print(f"Total params: {total:,}")
    print(f"Trainable params: {trainable:,}")

    for epoch in range(start_epoch, args.epochs):
        model.train()
        for step, batch in enumerate(tqdm(train_loader)):
            clean, reverb, lengths = batch['clean_vae'], batch['reverb_vae'], batch['length']
            clean = clean.to(accelerator.device)
            reverb = reverb.to(accelerator.device)
            lengths = lengths.to(accelerator.device)

            # Flow Matching noise mix
            noise = torch.randn_like(clean)
            sigmas = torch.rand((clean.shape[0],), dtype=clean.dtype, device=clean.device)
            timesteps = sigmas * 1000
            while len(sigmas.shape) < clean.ndim:
                sigmas = sigmas.unsqueeze(-1)
            noisy_target = sigmas * noise + (1.0 - sigmas) * clean
            velocity = noise - clean

            pred, pred_mask = model(x=noisy_target, timesteps=timesteps, mixture=reverb, x_len=lengths)

            loss = masked_mse(pred, velocity if args.v_prediction else noise, pred_mask)

            if not torch.isnan(loss):
                accelerator.backward(loss)

                # --- DEBUG 2: 看看 MoLEx 参数有没有梯度 ---
                if accelerator.is_main_process and (global_step % 100 == 0):
                    total_norm = 0.0
                    count = 0
                    for name, p in model.named_parameters():
                        if p.requires_grad and p.grad is not None:
                            if "lora_A" in name or "lora_B" in name or "gate" in name:
                                param_norm = p.grad.data.norm().item()
                                total_norm += param_norm
                                count += 1
                    if count > 0:
                        print(f"[DEBUG] step {global_step}: avg MoLEx grad norm = {total_norm / count:.6e}")
                    else:
                        print(f"[DEBUG] step {global_step}: no MoLEx grads found")


                optimizer.step()
                optimizer.zero_grad()

                global_step += 1
                losses += loss.item()

                if accelerator.is_main_process and global_step % args.log_step == 0:
                    with open(os.path.join(args.log_dir, 'log.txt'), 'a') as f:
                        f.write(time.asctime(time.localtime(time.time())) + '\n')
                        f.write('Epoch: [{}][{}]    Batch: [{}][{}]    Loss: {:.6f}\n'.format(
                            epoch + 1, args.epochs, step + 1, len(train_loader), losses / args.log_step))
                    losses = 0.0
            else:
                torch.cuda.empty_cache()
                if accelerator.is_main_process:
                    with open(os.path.join(args.log_dir, 'log.txt'), 'a') as f:
                        f.write(time.asctime(time.localtime(time.time())) + '\n')
                        f.write('Epoch: [{}][{}]    Batch: [{}][{}]  Nan  Loss\n'.format(
                            epoch + 1, args.epochs, step + 1, len(train_loader)))

        # ===== 保存 =====
        if accelerator.is_main_process and ((epoch + 1) % args.save_every == 0):
            accelerator.wait_for_everyone()
            unwrapped = accelerator.unwrap_model(model)

            save_obj = {
                "model": unwrapped.state_dict(),
                "optimizer": optimizer.state_dict(),  # 便于 resume（MoLEx-only param_groups）
                "epoch": epoch,
                "global_step": global_step,
                "use_molex": args.use_molex,
                "molex_rank": args.molex_rank,
                "molex_alpha": args.molex_alpha,
                "molex_num_experts": args.molex_num_experts,
                "molex_top_k": args.molex_top_k,
                "molex_noise_std": args.molex_noise_std,
                "molex_targets": args.molex_targets,
            }
            torch.save(save_obj, os.path.join(args.save_dir, f"{epoch}.pt"))

        accelerator.wait_for_everyone()


if __name__ == '__main__':
    main()
