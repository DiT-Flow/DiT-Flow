import yaml
import random
import argparse
import os
import time
import re
from typing import Iterable, Tuple

import torch
import torch.nn as nn
import torchaudio
import pandas as pd
import csv, random
from tqdm import tqdm
from torch.utils.data import DataLoader

from accelerate import Accelerator
from diffusers import FlowMatchEulerDiscreteScheduler

# ===== 你的项目依赖 =====
from model.solospeech.conditioners import SoloSpeech_TSR
from dataset import TSEDataset2

# ---------------------------
# ======== LoRA utils ========
# ---------------------------
class LoRALinear(nn.Module):
    """
    Linear with LoRA: W := W0 + scaling * (B @ A)
    Only A/B are trainable; W0 (weight/bias) are frozen.
    """
    def __init__(self, in_features, out_features, r=8, alpha=16, bias=True,
                 base_weight: torch.Tensor = None, base_bias: torch.Tensor = None,
                 init_zero_B: bool = True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.r = r
        self.alpha = alpha
        self.scaling = alpha / max(1, r)

        self.weight = nn.Parameter(torch.empty(out_features, in_features), requires_grad=False)
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features), requires_grad=False)
        else:
            self.register_parameter("bias", None)

        if base_weight is None:
            nn.init.kaiming_uniform_(self.weight, a=5**0.5)
        else:
            with torch.no_grad():
                self.weight.copy_(base_weight)
        if base_bias is not None and self.bias is not None:
            with torch.no_grad():
                self.bias.copy_(base_bias)

        if r > 0:
            self.A = nn.Parameter(torch.empty(r, in_features))
            self.B = nn.Parameter(torch.empty(out_features, r))
            nn.init.kaiming_uniform_(self.A, a=5**0.5)
            if init_zero_B:
                nn.init.zeros_(self.B)
            else:
                nn.init.kaiming_uniform_(self.B, a=5**0.5)
        else:
            self.register_parameter("A", None)
            self.register_parameter("B", None)
        self.merged = False

    def forward(self, x):
        out = torch.nn.functional.linear(x, self.weight, self.bias)
        if self.r and not self.merged:
            out = out + self.scaling * torch.nn.functional.linear(
                torch.nn.functional.linear(x, self.A, None), self.B, None
            )
        return out

    @torch.no_grad()
    def merge_lora(self):
        if self.r and not self.merged:
            delta = (self.B @ self.A) * self.scaling
            self.weight.add_(delta)
            self.merged = True

    @torch.no_grad()
    def unmerge_lora(self):
        if self.r and self.merged:
            delta = (self.B @ self.A) * self.scaling
            self.weight.sub_(delta)
            self.merged = False


def _replace_linear_with_lora(parent: nn.Module, name: str, r: int, alpha: int):
    old: nn.Linear = getattr(parent, name)
    new = LoRALinear(
        in_features=old.in_features,
        out_features=old.out_features,
        r=r, alpha=alpha,
        bias=(old.bias is not None),
        base_weight=old.weight.data.clone(),
        base_bias=None if old.bias is None else old.bias.data.clone(),
    )
    setattr(parent, name, new)

def apply_lora(model: nn.Module,
               r: int = 8,
               alpha: int = 16,
               target_keywords: Iterable[str] = (
                #    # attention
                #    "to_q", "to_k", "to_v", "q_proj", "k_proj", "v_proj",
                #    "proj", "out_proj", "qkv",
                #    # cross_attention
                #    "cross_attn.q", "cross_attn.k", "cross_attn.v", "cross_attn.proj",
                #    # mlp
                #    "mlp.fc1", "mlp.fc2", "mlp.linear1", "mlp.linear2"
                # attention
                # "SoloSpeech_TSR.UDiT.DiTBlock.attn.to_q", "SoloSpeech_TSR.UDiT.DiTBlock.attn.to_k", "SoloSpeech_TSR.UDiT.DiTBlock.attn.to_v"
                "mlp.w1", "mlp.w2"
               ),
               use_regex: bool = False) -> Tuple[int, int]:
    """
    Replace matched nn.Linear with LoRALinear.
    """
    hit, replaced = 0, 0

    def name_matches(full_name: str) -> bool:
        if use_regex:
            return any(re.search(p, full_name) for p in target_keywords)
        return any(k in full_name for k in target_keywords)

    for full_name, module in list(model.named_modules()):
        for child_name, child in list(module.named_children()):
            if isinstance(child, nn.Linear) and name_matches(f"{full_name}.{child_name}"):
                hit += 1
                _replace_linear_with_lora(module, child_name, r, alpha)
                replaced += 1
    return hit, replaced

def only_optimize_lora(model: nn.Module):
    """冻结除 LoRA 以外的参数"""
    for n, p in model.named_parameters():
        p.requires_grad_(("A" in n or "B" in n))

def lora_merge(model: nn.Module):
    for m in model.modules():
        if isinstance(m, LoRALinear):
            m.merge_lora()

def lora_unmerge(model: nn.Module):
    for m in model.modules():
        if isinstance(m, LoRALinear):
            m.unmerge_lora()

def safe_load_optimizer(optimizer: torch.optim.Optimizer, ckpt_opt_state: dict,
                        keep_hparams: bool = True, lr: float = None):
    """
    仅加载 state（动量等），保留当前 param_groups（从而保留“只训练 LoRA”和你设置的超参）。
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



def count_parameters(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable



# ---------------------------
# ====== /LoRA utils ========
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
    parser.add_argument('--diffusion-config', type=str, default='./config/base_solospeech_tse_cfm_urgent2026_lora.yaml')
    parser.add_argument('--autoencoder-path', type=str, default='./pretrained_models/audio-vae.pt')

    # 启动方式
    parser.add_argument('--resume-from', type=str, default=None, help='Resume training from a training checkpoint (model + optimizer + progress if available)')
    parser.add_argument('--pretrained', type=str, default="/export/fs05/tcao7/enhance/SoloSpeech/solospeech/base_solospeech_tse_cfm_urgent2026_baseline_full_ckpt/pretrained.pt", help='Initialize from a pretrained base model (model-only), then start a fresh LoRA finetune')

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

    # ===== LoRA =====
    parser.add_argument('--use-lora', action='store_true', help='Enable LoRA fine-tuning')
    parser.add_argument('--lora-r', type=int, default=8)
    parser.add_argument('--lora-alpha', type=int, default=16)
    parser.add_argument('--lora-targets', type=str,
                        default='mlp.w1,mlp.w2',
                        help='Comma-separated substrings or regex to match Linear layers')
    parser.add_argument('--lora-targets-regex', action='store_true',
                        help='Interpret --lora-targets as regex patterns')
    parser.add_argument('--lora-merge-on-save', action='store_true',
                        help='Merge LoRA into base weights before saving (for inference)')
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
    # args.log_dir = args.log_dir.replace('log', args.diff_config["system"] + '_log')
    # args.save_dir = args.save_dir.replace('ckpt', args.diff_config["system"] + '_ckpt')
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

    # 读取元数据、过滤（select augmentation == none）
    # meta_path = "/export/fs05/tcao7/urgent2026/simulation_train/log/meta.tsv"
    # df = pd.read_csv(meta_path, sep="\t", dtype=str)
    # df["fs"] = pd.to_numeric(df["fs"], errors="coerce")
    # df["length"] = pd.to_numeric(df["length"], errors="coerce")
    # df["duration_s"] = df["length"] / df["fs"]
    # mask = (df["duration_s"] < 20.0) & (df["augmentation"].fillna("").str.strip().str.lower() == "none")
    # noisy_paths = sorted(df.loc[mask, "noisy_path"].dropna().str.replace("simulation_train", "simulation_train_resample").tolist())
    # clean_paths = sorted(df.loc[mask, "clean_path"].dropna().str.replace("simulation_train", "simulation_train_resample").tolist())
    # print(f"kept {len(noisy_paths)} noisy files and {len(clean_paths)} clean files")

    META_TSV = "/export/fs05/tcao7/urgent2026/simulation_train/log/meta.tsv"
    TARGET_HOURS = 30.0             # select 12-hour data to finetune
    TOLERANCE_SECONDS = 300        # how close to target to aim for (±5 min)
    SEED = 2026                    # change for a different random selection

    target_seconds = TARGET_HOURS * 3600.0
    candidates = load_candidates(META_TSV)

    if not candidates:
        raise SystemExit("No candidates found with duration<20s and augmentation!='none'.")

    selected, total = random_pack(candidates, target_seconds, TOLERANCE_SECONDS, SEED)

    noisy_paths = sorted([s["noisy_path"] for s in selected if s["noisy_path"]])
    noisy_paths = [item.replace("simulation_train", "simulation_train_resample") for item in noisy_paths]
    clean_paths = sorted([s["clean_path"] for s in selected if s["clean_path"]])
    clean_paths = [item.replace("simulation_train", "simulation_train_resample") for item in clean_paths]









    train_set = TSEDataset2(reverb_dir=noisy_paths, clean_dir=clean_paths, debug=args.debug)
    train_loader = DataLoader(train_set, num_workers=args.num_workers, batch_size=args.batch_size,
                              shuffle=True, pin_memory=True, collate_fn=train_set.collate)

    accelerator = Accelerator(mixed_precision=args.amp)

    # ===== 1) 构建模型 =====
    model = SoloSpeech_TSR(args.diff_config['diffwrap']['UDiT'])
    total = sum(p.nelement() for p in model.parameters())
    print("Number of parameter: %.2fM" % (total / 1e6))

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

    # ===== 3) 注入 LoRA（在 prepare 之前） =====
    if args.use_lora:
        targets = tuple([s.strip() for s in args.lora_targets.split(",") if s.strip()])
        hit, replaced = apply_lora(
            model, r=args.lora_r, alpha=args.lora_alpha,
            target_keywords=targets, use_regex=args.lora_targets_regex
        )
        print(f"[LoRA] matched {hit} Linear layers, replaced {replaced}")
        only_optimize_lora(model)  # 冻结非 LoRA

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
        # 注意：如果你是先加载了 pretrained，这里会再覆盖一遍；resume 优先级更高是合理的
        missing, unexpected = model.load_state_dict(ckpt["model"], strict=False)
        if missing:
            print(f"[Resume] Missing keys (often due to LoRA adding new A/B): {len(missing)}")

        # 恢复优化器动量，但保留当前 param_groups 与超参（LoRA-only）
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

            # 可选：保存前合并 LoRA 方便推理
            did_merge = False
            if args.use_lora and args.lora_merge_on_save:
                lora_merge(unwrapped)
                did_merge = True

            save_obj = {
                "model": unwrapped.state_dict(),
                "optimizer": optimizer.state_dict(),  # 便于 resume（LoRA-only param_groups）
                "epoch": epoch,
                "global_step": global_step,
                "use_lora": args.use_lora,
                "lora_r": args.lora_r,
                "lora_alpha": args.lora_alpha,
                "lora_targets": args.lora_targets,
            }
            torch.save(save_obj, os.path.join(args.save_dir, f"{epoch}.pt"))

            # 如果后续还要继续训练且刚刚 merge 过，需要拆回
            if did_merge:
                lora_unmerge(unwrapped)

        accelerator.wait_for_everyone()


if __name__ == '__main__':
    main()
