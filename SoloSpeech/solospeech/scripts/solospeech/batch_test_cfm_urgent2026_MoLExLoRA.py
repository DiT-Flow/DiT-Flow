import yaml
import random
import argparse
import os
import re
import glob

import torch
import torch.nn as nn
import torch.nn.functional as F
import librosa
from tqdm import tqdm
from diffusers import FlowMatchEulerDiscreteScheduler

from model.solospeech.conditioners import SoloSpeech_TSR
from utils import save_audio
from vae_modules.autoencoder_wrapper import Autoencoder


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
    target_keywords=("mlp.w1", "mlp.w2"),
    use_regex: bool = False,
):
    """
    遍历模型，把匹配的 nn.Linear 替换成 MoLExLinear。

    默认匹配 DiTBlock.mlp 内的 w1/w2（名字包含 "mlp.w1" 或 "mlp.w2"）。
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


# ==========================
# ===== 原推理脚本主体 =====
# ==========================

parser = argparse.ArgumentParser()
parser.add_argument('--device', type=str, default='cuda')
parser.add_argument('--save-dir', type=str, default='/export/fs05/tcao7/enhance/SoloSpeech/solospeech/30hour/base_solospeech_tse_cfm_urgent2026_MoLExLoRA_topK_3_r_8_alpha_16_lr_e-4_30h_epoch199_denoised/')
parser.add_argument('--input_dir', type=str, default='/export/fs05/tcao7/urgent2026/simulation_validation_resample/noisy/0/')

# pre-trained model path
parser.add_argument('--autoencoder-path', type=str, default='/export/fs05/hwang258/SoloSpeech/pretrained_models/vae-200k.ckpt')
parser.add_argument('--eta', type=int, default=0)

parser.add_argument("--num_infer_steps", type=int, default=200)

# model configs
parser.add_argument('--vae-config', type=str, default='/export/fs05/hwang258/SoloSpeech/pretrained_models/config.json')
parser.add_argument('--tsr-config', type=str, default='./config/SoloSpeech-tse-base3-cfm.yaml')
parser.add_argument('--tsr-ckpt', type=str, default='/export/fs05/tcao7/enhance/SoloSpeech/solospeech/30hour/base_solospeech_tse_cfm_urgent2026_MoLExLoRA_topK_3_r_8_alpha_16_lr_e-4_30h_ckpt/199.pt')
parser.add_argument('--sample-rate', type=int, default=16000)
parser.add_argument('--debug', type=bool, default=False)

# log and random seed
parser.add_argument('--random-seed', type=int, default=2025)
args = parser.parse_args()

with open(args.tsr_config, 'r') as fp:
    args.tsr_config = yaml.safe_load(fp)

args.v_prediction = args.tsr_config["ddim"]["v_prediction"]


@torch.no_grad()
def sample_diffusion(tsr_model, autoencoder, std, scheduler, device,
                     mixture=None, lengths=None,
                     ddim_steps=50, eta=0, seed=2023):

    generator = torch.Generator(device=device).manual_seed(seed)
    scheduler.set_timesteps(ddim_steps)
    tsr_pred = torch.randn(mixture.shape, generator=generator, device=device)

    for t in scheduler.timesteps:
        model_output, _ = tsr_model(
            x=tsr_pred,
            timesteps=t,
            mixture=mixture,
            x_len=lengths,
        )
        tsr_pred = scheduler.step(model_output=model_output, timestep=t, sample=tsr_pred).prev_sample

    tsr_pred = autoencoder(embedding=tsr_pred.transpose(2, 1), std=std).squeeze(1)
    return tsr_pred


if __name__ == '__main__':

    os.makedirs(args.save_dir, exist_ok=True)

    # ===== Autoencoder =====
    autoencoder = Autoencoder(
        args.autoencoder_path,
        args.vae_config,
        'stft_vae',
        quantization_first=True
    )
    autoencoder.eval()
    autoencoder.to(args.device)

    # ===== 构建 TSR 模型（先在 CPU 上） =====
    tsr_model = SoloSpeech_TSR(
        args.tsr_config['diffwrap']['UDiT']
    )

    # ===== 读取 MoLEx ckpt 并决定是否注入 MoLEx =====
    ckpt_raw = torch.load(args.tsr_ckpt, map_location='cpu')

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
    if missing:
        print(f"[TSR] Missing keys when loading state_dict: {len(missing)}")
        if args.debug:
            for k in missing:
                print("  missing:", k)
    if unexpected:
        print(f"[TSR] Unexpected keys when loading state_dict: {len(unexpected)}")
        if args.debug:
            for k in unexpected:
                print("  unexpected:", k)

    tsr_model.to(args.device)
    tsr_model.eval()

    total = sum([param.nelement() for param in tsr_model.parameters()])
    print("TSR Number of parameters: %.2fM" % (total / 1e6))

    noise_scheduler = FlowMatchEulerDiscreteScheduler(**args.tsr_config["ddim"]['diffusers'])

    # ===== 读取输入音频 =====
    # 如果你要跑整个目录，用 glob：
    input_audios = glob.glob(os.path.join(args.input_dir, "*.flac"))
    input_audios = sorted(input_audios)

    # 如果你只想测试单个文件，可以改成：
    # input_audios = ['/export/fs05/tcao7/enhance/SoloSpeech/solospeech/672-2830-4446_2_reverb.wav']

    for input_audio in tqdm(input_audios):
        savename = os.path.basename(input_audio)
        mixture, _ = librosa.load(input_audio, sr=args.sample_rate)

        with torch.no_grad():
            mixture_input = torch.tensor(mixture).unsqueeze(0).to(args.device)  # [1, T]
            mixture_wav = mixture_input
            mixture_input, std = autoencoder(audio=mixture_input.unsqueeze(1))  # [1, 1, T] -> latent
            lengths = torch.LongTensor([mixture_input.shape[-1]]).to(args.device)

            tsr_pred = sample_diffusion(
                tsr_model,
                autoencoder,
                std,
                noise_scheduler,
                args.device,
                mixture=mixture_input.transpose(2, 1),
                lengths=lengths,
                ddim_steps=args.num_infer_steps,
                eta=args.eta,
                seed=args.random_seed,
            )

            out_path = os.path.join(args.save_dir, savename)
            save_audio(out_path, args.sample_rate, tsr_pred)
