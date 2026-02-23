import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from collections import defaultdict

from model.solospeech.conditioners import SoloSpeech_TSR
from dataset import DistortionEvalDataset  # 需要你自己实现
import pandas as pd

from typing import Iterable, Tuple
import re

# ==========================
# ===== MoLEx for Inference
# ==========================

class MoLExLinear(nn.Module):
    """
    Linear + Mixture of LoRA Experts (MoLEx)

    y = x W^T + scaling * sum_{i in TopK(x)} g_i(x) * (B_i A_i x)

    推理不需要 LB loss 和统计，只保留 forward 逻辑，
    这样可以兼容训练时保存的 MoLEx 参数。
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        r: int = 8,
        num_experts: int = 5,
        k: int = 3,
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

        # LoRA：A 随机，B 置零（保证 ΔW 初始≈0 但有梯度空间）
        nn.init.kaiming_uniform_(self.lora_A, a=5 ** 0.5)
        nn.init.zeros_(self.lora_B)

        # gate：0 初始化没问题
        nn.init.zeros_(self.gate.weight)
        if self.gate.bias is not None:
            nn.init.zeros_(self.gate.bias)

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

        # 2) gating （推理一般不加噪声；如果你想和训练完全对齐，可以保留）
        logits = self.gate(x_flat)  # [N, E]
        if self.noise_std > 0.0 and self.training:
            noise = torch.randn_like(logits) * self.noise_std
            logits = logits + noise

        # full softmax 概率（内部用不到，只为保持结构一致）
        probs = F.softmax(logits, dim=-1)  # [N, E]

        # top-k 路由
        topk_logits, topk_indices = torch.topk(logits, self.k, dim=-1)  # [N, k]
        topk_gates = F.softmax(topk_logits, dim=-1)                     # [N, k]

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

# ========== 1. 加载 MoLEx 模型 ==========

def load_molex_model(ckpt_path, diffusion_config):
    """
    ckpt_path: 你 finetune 后保存的 MoLEx 模型路径（.pt）
    diffusion_config: 和训练时一样的 config dict（可以直接复用 yaml）
    """
    tsr_model = SoloSpeech_TSR(
        diffusion_config['diffwrap']['UDiT']
    )

    # ===== 读取 MoLEx ckpt 并决定是否注入 MoLEx =====
    ckpt_raw = torch.load(ckpt_path, map_location='cpu')

    if isinstance(ckpt_raw, dict) and "model" in ckpt_raw:
        state_dict = ckpt_raw["model"]
        meta = ckpt_raw
    else:
        state_dict = ckpt_raw
        meta = {}

    use_molex = meta.get("use_molex", False)

    if use_molex:
        print("[MoLEx] Detected MoLEx-LoRA checkpoint. Injecting MoLEx layers for inference...")
        r = meta.get("molex_rank", 8)
        alpha = meta.get("molex_alpha", 16.0)
        num_experts = meta.get("molex_num_experts", 5)
        top_k = meta.get("molex_top_k", 3)
        noise_std = meta.get("molex_noise_std", 0.0)  # 推理时一般设 0，更稳；如需完全对齐训练可调成 1.0
        targets = meta.get("molex_targets", "mlp.w1,mlp.w2")
        use_regex = meta.get("molex_targets_regex", False)

        target_keywords = tuple([s.strip() for s in targets.split(",") if s.strip()])
        hit, replaced = apply_molex(
            tsr_model,
            r=r,
            alpha=alpha,
            num_experts=num_experts,
            k=top_k,
            noise_std=noise_std,
            target_keywords=target_keywords,
            use_regex=use_regex,
        )
        print(f"[MoLEx] matched {hit} Linear layers, replaced {replaced}")
    else:
        print("[MoLEx] use_molex=False in checkpoint (or not found). Using base TSR model only.")

    # ===== 加载权重 =====
    missing, unexpected = tsr_model.load_state_dict(state_dict, strict=False)
    # if missing:
    #     print(f"[TSR] Missing keys when loading state_dict: {len(missing)}")
    #     if args.debug:
    #         for k in missing:
    #             print("  missing:", k)
    # if unexpected:
    #     print(f"[TSR] Unexpected keys when loading state_dict: {len(unexpected)}")
    #     if args.debug:
    #         for k in unexpected:
    #             print("  unexpected:", k)

    tsr_model.to('cuda')
    tsr_model.eval()

    return tsr_model


# ========== 2. 注册 hook，收集 gating 行为 ==========

def register_molex_hooks(model):
    """
    为所有 MoLExLinear 注册 forward hook，
    在一个全局字典里记录每一层的 gating probs（每个 sample 的平均）。
    返回 hooks 列表 + stats dict（后者方便外部访问）。
    """
    molex_stats = defaultdict(list)  # key: layer_name, value: list of [B, num_experts] tensors
    hooks = []

    def make_hook(name):
        def hook(module, inputs, output):
            """
            inputs: (x, ), x: [B, T, D] 或 [B, D]
            output: y, 同形状
            我们用 module.gate 自己算 logits -> softmax，得到 gating probs。
            """
            x = inputs[0]  # [B, ..., D]
            B = x.shape[0]
            D = x.shape[-1]

            x_flat = x.reshape(-1, D)                # [N, D]
            logits = module.gate(x_flat)             # [N, E]
            probs = F.softmax(logits, dim=-1)        # [N, E]
            probs = probs.view(B, -1, module.num_experts)  # [B, T, E]
            probs_mean = probs.mean(dim=1)           # [B, E]，每个样本在这一层的平均 gating

            molex_stats[name].append(probs_mean.detach().cpu())

        return hook

    for name, module in model.named_modules():
        if isinstance(module, MoLExLinear):
            h = module.register_forward_hook(make_hook(name))
            hooks.append(h)

    return hooks, molex_stats


# ========== 3. 在带失真标签的数据上跑一遍，聚合统计 ==========

def analyze_expert_routing(model, dataloader, num_distortions):
    """
    model: MoLEx 模型
    dataloader: batch 里必须有 'reverb_vae', 'length', 'distortion_id'
    num_distortions: 失真类型个数（你说有 5 种 unseen distortions → 5）

    返回：
      per_layer_stats: dict[layer_name] -> {
          "mean_probs": [num_distortions, num_experts] tensor,
          "counts": [num_distortions] 样本数
      }
    """
    hooks, molex_stats = register_molex_hooks(model)

    # 每个 layer、每个 distortion_id 累加 gating probs
    per_layer_sum = {}    # layer -> [num_distortions, num_experts]
    per_layer_count = {}  # layer -> [num_distortions]

    with torch.no_grad():
        for batch in dataloader:
            reverb = batch['noisy_vae'].cuda()   # 或者你用 noisy_vae
            lengths = batch['length'].cuda()
            dist_ids = batch['distortion_id'].cuda()  # [B]，0..num_distortions-1

            # 这里假设你 inference 只需要 reverb + lengths + timesteps，
            # timesteps 可以先随便给个固定值或 0（分析 gating 不太依赖 time）
            B = reverb.shape[0]
            timesteps = torch.zeros(B, device=reverb.device)

            # forward 一次，把 hooks 里的 probs 填满
            _ = model(x=reverb, timesteps=timesteps, mixture=reverb, x_len=lengths)

            # 现在 molex_stats[layer_name] 里多 append 了一次 [B, E]
            for layer_name, probs_list in molex_stats.items():
                # 取出这次 forward 新 append 的那一块
                probs_batch = probs_list[-1]    # [B, E]
                E = probs_batch.shape[-1]

                if layer_name not in per_layer_sum:
                    per_layer_sum[layer_name] = torch.zeros(num_distortions, E)
                    per_layer_count[layer_name] = torch.zeros(num_distortions)

                for b in range(B):
                    d = int(dist_ids[b].item())
                    per_layer_sum[layer_name][d] += probs_batch[b]
                    per_layer_count[layer_name][d] += 1

    # 计算平均 probs
    per_layer_stats = {}
    for layer_name, sum_mat in per_layer_sum.items():
        counts = per_layer_count[layer_name].unsqueeze(-1)  # [D, 1]
        # 避免除 0
        counts = torch.clamp(counts, min=1.0)
        mean_probs = sum_mat / counts   # [D, E]

        per_layer_stats[layer_name] = {
            "mean_probs": mean_probs,              # [num_distortions, num_experts]
            "counts": per_layer_count[layer_name]  # [num_distortions]
        }

    # 清理 hooks
    for h in hooks:
        h.remove()

    return per_layer_stats


# ========== 4. 可视化：heatmap ==========

def plot_expert_heatmap(mean_probs, distortion_names=None, title="", save_path=None):
    """
    mean_probs: [num_distortions, num_experts] tensor
    distortion_names: list[str] 长度 = num_distortions，用作 y 轴标签
    """
    import numpy as np

    probs = mean_probs.numpy()
    D, E = probs.shape

    plt.figure(figsize=(1.5 * E, 1.0 * D + 2))
    plt.imshow(probs, aspect='auto')
    plt.colorbar(label="mean gate probability")

    plt.xlabel("Expert id")
    plt.ylabel("Distortion type")

    plt.xticks(range(E), [str(e) for e in range(E)])
    if distortion_names is not None and len(distortion_names) == D:
        plt.yticks(range(D), distortion_names)
    else:
        plt.yticks(range(D), [f"D{d}" for d in range(D)])

    plt.title(title)
    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path, dpi=200)
    plt.show()


# ========== 5. 一个 main 示例，把上述串起来 ==========

def main():
    import yaml
    from torch.utils.data import DataLoader

    # ---- 1) 读 diffusion config（和训练时一样） ----
    with open("./config/base_solospeech_tse_cfm_urgent2026_lora.yaml", "r") as f:
        diff_config = yaml.safe_load(f)

    # ---- 2) 加载 MoLEx finetune 后的 checkpoint ----
    ckpt_path = "/export/fs05/tcao7/enhance/SoloSpeech/solospeech/30hour/base_solospeech_tse_cfm_urgent2026_MoLExLoRA_topK_3_r_8_alpha_16_lr_e-4_loss_switch_coef_e-2_loss_orth_coef_1.0_attn_30h_ckpt/199.pt"   # TODO: 换成你的路径
    model = load_molex_model(ckpt_path, diff_config)

    # print(model)

    # def debug_model_molex_presence(model):
    #     name_hits = []
    #     type_hits = []
    #     for n, m in model.named_modules():
    #         if "mlp.w1" in n or "mlp.w2" in n:
    #             name_hits.append((n, type(m).__name__, m.__class__.__module__))
    #         # count by attribute presence rather than isinstance
    #         if hasattr(m, "lora_A") and hasattr(m, "lora_B") and hasattr(m, "gate"):
    #             type_hits.append((n, type(m).__name__, m.__class__.__module__))

    #     print(f"[DEBUG] modules with name containing mlp.w1/w2: {len(name_hits)}")
    #     print(f"[DEBUG] modules that LOOK LIKE MoLEx (have lora_A/lora_B/gate): {len(type_hits)}")

    #     if name_hits[:10]:
    #         print("[DEBUG] first few mlp.w* modules:")
    #         for x in name_hits[:10]:
    #             print("   ", x)

    #     if type_hits[:10]:
    #         print("[DEBUG] first few MoLEx-like modules:")
    #         for x in type_hits[:10]:
    #             print("   ", x)

    #     # parameter-level sanity check
    #     pnames = [n for n, _ in model.named_parameters()]
    #     molex_params = [n for n in pnames if ("lora_A" in n or "lora_B" in n or ".gate." in n)]
    #     print(f"[DEBUG] parameters containing lora_A/lora_B/gate: {len(molex_params)}")
    #     if molex_params[:10]:
    #         print("[DEBUG] first few MoLEx params:")
    #         for n in molex_params[:10]:
    #             print("   ", n)


    # # call it after loading weights
    # debug_model_molex_presence(model)


    # ---- 3) 构建 eval dataloader ----
    # 你需要实现 build_eval_dataloader：
    #   - 每个 batch 返回:
    #       batch["reverb_vae"]: [B, ...]
    #       batch["length"]: [B]
    #       batch["distortion_id"]: [B], e.g. 0..4

    # train_set = TSEDataset2(
    #     reverb_dir=noisy_paths, 
    #     clean_dir=clean_paths,
    #     debug=args.debug,
    # )
    # train_loader = DataLoader(train_set, num_workers=args.num_workers, batch_size=args.batch_size, shuffle=True, pin_memory=True, collate_fn=train_set.collate)


    ######################### loading data #########################

    meta_path = "/export/fs05/tcao7/urgent2026/simulation_validation/log/meta.tsv"
    df = pd.read_csv(meta_path, sep="\t", dtype=str)

    # ---- duration ----
    df["fs"] = pd.to_numeric(df["fs"], errors="coerce")
    df["length"] = pd.to_numeric(df["length"], errors="coerce")
    df["duration_s"] = df["length"] / df["fs"]

    # ---- distortion parsing ----
    DISTORTIONS = ("clipping", "codec", "packet_loss", "bandwidth", "wind_noise")

    def distortion_families(aug: str):
        """Return sorted list of unique distortion families found in augmentation."""
        if pd.isna(aug):
            return []
        s = str(aug).strip().lower()
        if s == "" or s == "none":
            return []

        parts = [p.strip() for p in s.split("/") if p.strip()]
        families = set()

        for p in parts:
            # normalize bandwidth naming variants (per your examples)
            if p.startswith("bandwidth_limitation") or p.startswith("bandwidth"):
                families.add("bandwidth")
                continue

            for d in DISTORTIONS:
                if p.startswith(d):
                    families.add(d)
                    break

        return sorted(families)

    df["families"] = df["augmentation"].apply(distortion_families)
    df["n_distortions"] = df["families"].apply(len)

    # label rows with 0 distortions as "none"
    df["combo"] = df["families"].apply(lambda x: "none" if len(x) == 0 else "+".join(x))

    # single family (including "none" only when n=0)
    df["single_family"] = df["families"].apply(lambda x: x[0] if len(x) == 1 else pd.NA)

    # ---- (optional) apply duration filter here ----
    # If you want duration < 20s for everything below, set USE_DURATION_FILTER=True.
    USE_DURATION_FILTER = False

    if USE_DURATION_FILTER:
        base = df[df["duration_s"] < 20.0].copy()
    else:
        base = df.copy()

    # ---- COUNT TABLES ----
    # 1) Count by number of distortions (0/1/2/3)
    counts_n = base["n_distortions"].value_counts().sort_index()

    print("\nCounts by number of distortions (n_distortions):")
    print(counts_n.to_string())

    # 2) Count by single distortion family (exactly one) + optional none (0)
    counts_single = (
        base.assign(single_or_none=base["single_family"].fillna(pd.NA))
            .loc[base["n_distortions"].isin([0, 1])]
    )
    counts_single["single_or_none"] = counts_single.apply(
        lambda r: "none" if r["n_distortions"] == 0 else r["single_family"],
        axis=1
    )
    counts_single_table = counts_single["single_or_none"].value_counts().reindex(
        ["none"] + list(DISTORTIONS), fill_value=0
    )

    print("\nCounts for none (0 distortions) and single-family (1 distortion):")
    print(counts_single_table.to_string())

    # 3) Count by combination label (none, single, pairs, triples)
    counts_combo = base["combo"].value_counts()

    print("\nTop combination counts (combo):")
    print(counts_combo.head(30).to_string())

    # ---- PATH PAIRS PER TYPE (single distortion only) ----
    # Keep alignment: do NOT sort columns independently; sort rows once.
    pairs_base = base[["noisy_path", "clean_path", "n_distortions", "single_family", "combo"]].dropna(
        subset=["noisy_path", "clean_path"]
    ).copy()

    # optional path rewrite
    pairs_base["noisy_path"] = pairs_base["noisy_path"].str.replace(
        "simulation_validation", "simulation_validation_resample", regex=False
    )
    pairs_base["clean_path"] = pairs_base["clean_path"].str.replace(
        "simulation_validation", "simulation_validation_resample", regex=False
    )

    # Dictionary: {family: list of (noisy_path, clean_path)}
    pairs_by_single_family = {}
    for fam in DISTORTIONS:
        sub = pairs_base[(pairs_base["n_distortions"] == 1) & (pairs_base["single_family"] == fam)]
        sub = sub.sort_values(["noisy_path", "clean_path"]).reset_index(drop=True)
        pairs_by_single_family[fam] = list(zip(sub["noisy_path"].tolist(), sub["clean_path"].tolist()))

    print("\nPairs per SINGLE distortion family (duration-filtered if enabled):")
    for fam in DISTORTIONS:
        print(f"  {fam}: {len(pairs_by_single_family[fam])} pairs")

    # ---- (optional) PATH PAIRS PER COMBINATION (including none / multi) ----
    pairs_by_combo = {}
    for combo_label, sub in pairs_base.groupby("combo", sort=False):
        sub = sub.sort_values(["noisy_path", "clean_path"]).reset_index(drop=True)
        pairs_by_combo[combo_label] = list(zip(sub["noisy_path"].tolist(), sub["clean_path"].tolist()))

    print("\nPairs per COMBINATION label (showing top 15 combos by count):")
    top15 = counts_combo.head(15).index.tolist()
    for c in top15:
        print(f"  {c}: {len(pairs_by_combo.get(c, []))} pairs")
    
    from pathlib import Path

    def to_npz(p: str) -> str:
        return str(Path(p).with_suffix(".npz"))

    pairs_by_single_family_npz = {
        k: [(to_npz(noisy), to_npz(clean)) for noisy, clean in pairs]
        for k, pairs in pairs_by_single_family.items()
}





    eval_set = DistortionEvalDataset(
        distort_dict=pairs_by_single_family_npz,
        max_items_per_distortion=None,
        seed=2026,
    )
    eval_loader = DataLoader(
        eval_set,
        batch_size=8,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        collate_fn=eval_set.collate
    )



    num_distortions = 5

    # ---- 4) 做路由分析 ----
    per_layer_stats = analyze_expert_routing(model, eval_loader, num_distortions)

    print(per_layer_stats)

    # ---- 5) 画图：对每一层都画一个 heatmap ----
    distortion_names = [f"dist{d}" for d in range(num_distortions)]
    out_dir = "./molex_routing_plots"
    os.makedirs(out_dir, exist_ok=True)

    for layer_name, stats in per_layer_stats.items():
        mean_probs = stats["mean_probs"]  # [D, E]
        counts = stats["counts"]

        print(f"Layer: {layer_name}")
        print("  num samples per distortion:", counts.tolist())

        title = f"MoLEx routing @ {layer_name}"
        save_path = os.path.join(out_dir, f"{layer_name.replace('.', '_')}.png")
        plot_expert_heatmap(mean_probs, distortion_names, title=title, save_path=save_path)


if __name__ == "__main__":
    main()
