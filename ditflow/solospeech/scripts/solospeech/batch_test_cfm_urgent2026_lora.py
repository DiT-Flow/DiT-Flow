# -*- coding: utf-8 -*-
import yaml
import argparse
import os
import glob
import time
from typing import Iterable, Tuple
import re

import torch
import torch.nn as nn
import librosa
import pandas as pd  # 若不需要可去掉
from tqdm import tqdm
from diffusers import FlowMatchEulerDiscreteScheduler

from model.solospeech.conditioners import SoloSpeech_TSR
from vae_modules.autoencoder_wrapper import Autoencoder
from utils import save_audio


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
                   # attention
                   "to_q", "to_k", "to_v", "q_proj", "k_proj", "v_proj",
                   "proj", "out_proj", "qkv",
                   # cross_attention
                   "cross_attn.q", "cross_attn.k", "cross_attn.v", "cross_attn.proj",
                   # mlp
                   "mlp.fc1", "mlp.fc2", "mlp.linear1", "mlp.linear2"
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

def lora_merge(model: nn.Module):
    for m in model.modules():
        if isinstance(m, LoRALinear):
            m.merge_lora()

def lora_unmerge(model: nn.Module):
    for m in model.modules():
        if isinstance(m, LoRALinear):
            m.unmerge_lora()
# ---------------------------
# ====== /LoRA utils ========
# ---------------------------


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--save-dir', type=str, default='/export/fs05/tcao7/enhance/SoloSpeech/solospeech/30hour/base_solospeech_tse_cfm_urgent2026_lora_r_8_alpha_16_30h_epoch199_denoised/')
    parser.add_argument('--input_dir', type=str, default='/export/fs05/tcao7/urgent2026/simulation_validation_resample/noisy/0/')
    parser.add_argument('--sample-rate', type=int, default=16000)
    parser.add_argument('--debug', action='store_true')

    # VAE / model config
    parser.add_argument('--autoencoder-path', type=str, default='/export/fs05/hwang258/SoloSpeech/pretrained_models/vae-200k.ckpt')
    parser.add_argument('--vae-config', type=str, default='/export/fs05/hwang258/SoloSpeech/pretrained_models/config.json')
    parser.add_argument('--tsr-config', type=str, default='./config/SoloSpeech-tse-base3-cfm.yaml')
    parser.add_argument('--tsr-ckpt', type=str, default='/export/fs05/tcao7/enhance/SoloSpeech/solospeech/30hour/base_solospeech_tse_cfm_urgent2026_lora_r_8_alpha_16_30h_ckpt/199.pt')

    # sampler
    parser.add_argument("--num_infer_steps", type=int, default=200)
    parser.add_argument('--eta', type=int, default=0)
    parser.add_argument('--random-seed', type=int, default=2025)

    # ===== LoRA 开关（与训练脚本一致）=====
    parser.add_argument('--use-lora', action='store_true', help='Enable LoRA at inference (for un-merged LoRA checkpoints)')
    parser.add_argument('--lora-r', type=int, default=8)
    parser.add_argument('--lora-alpha', type=int, default=16)
    parser.add_argument('--lora-targets', type=str,
                        default='mlp.w1,mlp.w2',
                        help='Comma-separated substrings or regex to match Linear layers')
    parser.add_argument('--lora-targets-regex', action='store_true',
                        help='Interpret --lora-targets as regex patterns')
    parser.add_argument('--lora-merge-at-load', action='store_true',
                        help='After loading weights, merge LoRA into base for faster inference')

    # 可选：从基座 + 单独 LoRA adapter 的两段式加载
    parser.add_argument('--pretrained', type=str, default=None, help='Base model weights (model-only)')
    parser.add_argument('--lora-only-ckpt', type=str, default=None, help='LoRA adapter-only checkpoint to apply on top of --pretrained')
    return parser


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


def main():
    parser = build_parser()
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)

    # 读取 TSR 配置
    with open(args.tsr_config, 'r') as fp:
        args.tsr_config = yaml.safe_load(fp)
    args.v_prediction = args.tsr_config["ddim"]["v_prediction"]

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # ===== Autoencoder =====
    autoencoder = Autoencoder(args.autoencoder_path, args.vae_config, 'stft_vae', quantization_first=True)
    autoencoder.eval().to(device)

    # ===== 构建 TSR 模型 =====
    tsr_model = SoloSpeech_TSR(args.tsr_config['diffwrap']['UDiT']).to(device)

    # ===== 权重加载的几种场景 =====
    # 优先：单 ckpt 直接加载（可能包含 LoRA，也可能已合并）
    ckpt = None
    if args.tsr_ckpt and os.path.exists(args.tsr_ckpt):
        ckpt = torch.load(args.tsr_ckpt, map_location='cpu')
        state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt

        # 若 ckpt 记录了 LoRA 元信息，则默认开启 use_lora（可被命令行覆盖）
        ckpt_use_lora = bool(ckpt.get("use_lora", False)) if isinstance(ckpt, dict) else False
        if ckpt_use_lora and not args.use_lora:
            print("[Info] ckpt indicates LoRA was used; enabling --use-lora for loading...")
            args.use_lora = True
        if "lora_r" in ckpt and args.lora_r == parser.get_default("lora_r"):
            args.lora_r = int(ckpt["lora_r"])
        if "lora_alpha" in ckpt and args.lora_alpha == parser.get_default("lora_alpha"):
            args.lora_alpha = int(ckpt["lora_alpha"])
        if "lora_targets" in ckpt and args.lora_targets == parser.get_default("lora_targets"):
            args.lora_targets = ckpt["lora_targets"]

        # 如果需要 LoRA（未合并的情况），必须在 load_state_dict 之前注入
        if args.use_lora:
            targets = tuple([s.strip() for s in args.lora_targets.split(",") if s.strip()])
            hit, replaced = apply_lora(
                tsr_model, r=args.lora_r, alpha=args.lora_alpha,
                target_keywords=targets, use_regex=args.lora_targets_regex
            )
            print(f"[LoRA] matched {hit} Linear layers, replaced {replaced}")

        missing, unexpected = tsr_model.load_state_dict(state, strict=False)
        if missing:
            print(f"[Load] missing keys: {len(missing)} (expected if LoRA newly injected or ckpt merged)")
        if unexpected:
            print(f"[Load] unexpected keys: {len(unexpected)} (ckpt has extras not in current model)")

        # 推理前可选合并
        if args.use_lora and args.lora_merge_at_load:
            # 注意：命令行参数有连字符，转属性名需要替换或直接用 getattr 不太方便。
            # 这里我们改为从 args.__dict__ 中取：
            pass

    # 上面写 lora-merge-at-load 时属性名包含“-”，python 不能直接 args.lora-merge-at-load。
    # 改名为下划线版本，重新处理：
    args.__dict__['lora_merge_at_load'] = getattr(args, 'lora_merge_at_load', False) or args.__dict__.get('lora-merge-at-load', False)

    # 如果用户选择“两段式加载”：--pretrained + --lora-only-ckpt
    if args.pretrained and os.path.exists(args.pretrained):
        base = torch.load(args.pretrained, map_location='cpu')
        base_state = base["model"] if isinstance(base, dict) and "model" in base else base
        _ = tsr_model.load_state_dict(base_state, strict=False)
        print(f"[Init] loaded base weights from {args.pretrained}")

        if args.lora_only_ckpt and os.path.exists(args.lora_only_ckpt):
            # 注入 LoRA
            if not args.use_lora:
                args.use_lora = True
            targets = tuple([s.strip() for s in args.lora_targets.split(",") if s.strip()])
            hit, replaced = apply_lora(
                tsr_model, r=args.lora_r, alpha=args.lora_alpha,
                target_keywords=targets, use_regex=args.lora_targets_regex
            )
            print(f"[LoRA] matched {hit} Linear layers, replaced {replaced}")

            lora_ckpt = torch.load(args.lora_only_ckpt, map_location='cpu')
            lora_state = lora_ckpt["model"] if "model" in lora_ckpt else lora_ckpt
            _ = tsr_model.load_state_dict(lora_state, strict=False)
            print(f"[Init] applied LoRA adapter from {args.lora_only_ckpt}")

    # 若用户要求加载后合并（更快推理）
    if args.use_lora and args.__dict__.get('lora_merge_at_load', False):
        lora_merge(tsr_model)
        print("[LoRA] merged into base for inference")

    tsr_model.eval().to(device)

    total = sum([p.nelement() for p in tsr_model.parameters()])
    print("TSR Number of parameter: %.2fM" % (total / 1e6))

    # ===== Scheduler =====
    noise_scheduler = FlowMatchEulerDiscreteScheduler(**args.tsr_config["ddim"]['diffusers'])

    # ===== 输入音频列表 =====
    input_list = []
    if os.path.isdir(args.input_dir):
        input_list = sorted(glob.glob(os.path.join(args.input_dir, "*.flac")))
    elif os.path.isfile(args.input_dir):
        input_list = [args.input_dir]
    else:
        raise FileNotFoundError(f"input_dir not found: {args.input_dir}")

    # ===== 推理循环 =====
    for input_audio in tqdm(input_list):
        savename = os.path.basename(input_audio)
        mixture, _ = librosa.load(input_audio, sr=args.sample_rate)

        with torch.no_grad():
            mixture_input = torch.tensor(mixture, dtype=torch.float32).unsqueeze(0).to(device)  # [B=1, T]
            mixture_wav = mixture_input  # 若需要保存输入
            mixture_emb, std = autoencoder(audio=mixture_input.unsqueeze(1))  # -> [B, C, L]
            lengths = torch.LongTensor([mixture_emb.shape[-1]]).to(device)

            tsr_pred = sample_diffusion(
                tsr_model, autoencoder, std, noise_scheduler, device,
                mixture=mixture_emb.transpose(2, 1),  # [B, L, C]
                lengths=lengths,
                ddim_steps=args.num_infer_steps,
                eta=args.eta,
                seed=args.random_seed
            )

        out_path = os.path.join(args.save_dir, savename)
        save_audio(out_path, args.sample_rate, tsr_pred)
        # 如需对齐原始长度/保存中间结果，可在此扩展

    print("Done.")


if __name__ == '__main__':
    main()
