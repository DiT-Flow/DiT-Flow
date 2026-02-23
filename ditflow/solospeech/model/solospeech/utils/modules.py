import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint
from torch.cuda.amp import autocast
import math
import einops
from einops import rearrange, repeat
from inspect import isfunction
from .timm import trunc_normal_


# disable in checkpoint mode
# @torch.jit.script
def film_modulate(x, shift, scale):
    return x * (1 + scale) + shift


def timestep_embedding(timesteps, dim, max_period=10000):
    """
    Create sinusoidal timestep embeddings.

    :param timesteps: a 1-D Tensor of N indices, one per batch element.
                      These may be fractional.
    :param dim: the dimension of the output.
    :param max_period: controls the minimum frequency of the embeddings.
    :return: an [N x dim] Tensor of positional embeddings.
    """
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
    ).to(device=timesteps.device)
    args = timesteps[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding


class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """

    def __init__(self, hidden_size, frequency_embedding_size=256, 
                 out_size=None):
        super().__init__()
        if out_size is None:
            out_size = hidden_size
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, out_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    def forward(self, t):
        t_freq = timestep_embedding(t, self.frequency_embedding_size).type(
            self.mlp[0].weight.dtype)
        t_emb = self.mlp(t_freq)
        return t_emb


def patchify(imgs, patch_size, input_type='2d'):
    if input_type == '2d':
        x = einops.rearrange(imgs, 'B C (h p1) (w p2) -> B (h w) (p1 p2 C)', p1=patch_size, p2=patch_size)
    elif input_type == '1d':
        x = einops.rearrange(imgs, 'B C (h p1) -> B h (p1 C)', p1=patch_size)
    return x


def unpatchify(x, channels=3, input_type='2d', img_size=None):
    if input_type == '2d':
        patch_size = int((x.shape[2] // channels) ** 0.5)
        # h = w = int(x.shape[1] ** .5)
        h, w = img_size[0] // patch_size, img_size[1] // patch_size
        assert h * w == x.shape[1] and patch_size ** 2 * channels == x.shape[2]
        x = einops.rearrange(x, 'B (h w) (p1 p2 C) -> B C (h p1) (w p2)', h=h,
                             p1=patch_size, p2=patch_size)
    elif input_type == '1d':
        patch_size = int((x.shape[2] // channels))
        h = x.shape[1]
        assert patch_size * channels == x.shape[2]
        x = einops.rearrange(x, 'B h (p1 C) -> B C (h p1)', h=h, p1=patch_size)
    return x


class PatchEmbed(nn.Module):
    """
     Image to Patch Embedding
    """

    def __init__(self, patch_size, in_chans=3, embed_dim=768, input_type='2d'):
        super().__init__()
        self.patch_size = patch_size
        self.input_type = input_type
        if input_type == '2d':
            self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size, bias=True)
        elif input_type == '1d':
            self.proj = nn.Conv1d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size, bias=True)

    def forward(self, x):
        if self.input_type == '2d':
            B, C, H, W = x.shape
            assert H % self.patch_size == 0 and W % self.patch_size == 0
        elif self.input_type == '1d':
            B, C, H = x.shape
            assert H % self.patch_size == 0

        x = self.proj(x).flatten(2).transpose(1, 2)
        return x


class PositionalConvEmbedding(nn.Module):
    """
    Relative positional embedding used in HuBERT
    """

    def __init__(self, dim=768, kernel_size=128, groups=16):
        super().__init__()
        self.conv = nn.Conv1d(
            dim,
            dim,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            groups=groups,
            bias=True
        )
        self.conv = nn.utils.parametrizations.weight_norm(self.conv, name="weight", dim=2)

    def forward(self, x):
        # B C T
        x = self.conv(x)
        x = F.gelu(x[:, :, :-1])
        return x


class SinusoidalPositionalEncoding(nn.Module):
    def __init__(
        self,
        dim_model: int,
        dropout: float = 0.0,
        scale: bool = False,
        alpha: bool = True,
    ):
        super().__init__()
        self.dim_model = dim_model
        self.x_scale = math.sqrt(dim_model) if scale else 1.0
        self.alpha = nn.Parameter(torch.ones(1), requires_grad=alpha)
        if alpha:
            print('pe alpha is trainable')
        self.dropout = torch.nn.Dropout(p=dropout)

        self.reverse = False
        self.pe = None
        self.extend_pe(torch.tensor(0.0).expand(1, 4000))

    def extend_pe(self, x):
        """Reset the positional encodings."""
        if self.pe is not None:
            if self.pe.size(1) >= x.size(1):
                if self.pe.dtype != x.dtype or self.pe.device != x.device:
                    self.pe = self.pe.to(dtype=x.dtype, device=x.device)
                return
        pe = torch.zeros(x.size(1), self.dim_model)
        if self.reverse:
            position = torch.arange(
                x.size(1) - 1, -1, -1.0, dtype=torch.float32
            ).unsqueeze(1)
        else:
            position = torch.arange(
                0, x.size(1), dtype=torch.float32
            ).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, self.dim_model, 2, dtype=torch.float32)
            * -(math.log(10000.0) / self.dim_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.pe = pe.to(device=x.device, dtype=x.dtype).detach()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.extend_pe(x)
        output = x.unsqueeze(-1) if x.ndim == 2 else x
        output = output * self.x_scale + self.alpha * self.pe[:, : x.size(1)]
        return self.dropout(output)


class PE_wrapper(nn.Module):
    def __init__(self, dim=768, method='abs', length=None, **kwargs):
        super().__init__()
        self.method = method
        if method == 'abs':
            # init absolute pe like UViT
            self.length = length
            self.abs_pe = nn.Parameter(torch.zeros(1, length, dim))
            trunc_normal_(self.abs_pe, std=.02)
        elif method == 'conv':
            self.conv_pe = PositionalConvEmbedding(dim=dim, **kwargs)
        elif method == 'sinu':
            self.sinu_pe = SinusoidalPositionalEncoding(dim_model=dim, **kwargs)
        elif method == 'none':
            # skip pe
            self.id = nn.Identity()
        else:
            raise NotImplementedError

    def forward(self, x):
        if self.method == 'abs':
            _, L, _ = x.shape
            assert L <= self.length
            x = x + self.abs_pe[:, :L, :]
        elif self.method == 'conv':
            x = x + self.conv_pe(x)
        elif self.method == 'sinu':
            x = self.sinu_pe(x)
        elif self.method == 'none':
            x = self.id(x)
        else:
            raise NotImplementedError
        return x


class RMSNorm(torch.nn.Module):
    def __init__(self, hidden_size, eps=1e-6, device=None):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(hidden_size))

    def forward(self, x):
        x_fp32 = x.float()
        x_normed = x_fp32 * torch.rsqrt(x_fp32.pow(2).mean(-1, keepdim=True) + self.eps)
        return (x_normed * self.weight).type_as(x)


class FeedForward(nn.Module):
    def __init__(
        self,
        in_features: int,
        hidden_size: int,
        multiple_of: int,
        ffn_dim_multiplier=None,
        bias=True
    ):
        super().__init__()
        # keep parameter count and computation constant compared to standard FFN
        hidden_size = int(2 * hidden_size / 3)
        # custom dim factor multiplier
        if ffn_dim_multiplier is not None:
            hidden_size = int(ffn_dim_multiplier * hidden_size)
        hidden_size = multiple_of * ((hidden_size + multiple_of - 1) // multiple_of)

        self.hidden_dim = hidden_size
        self.w1 = nn.Linear(in_features, 2 * hidden_size, bias=bias)
        self.w2 = nn.Linear(hidden_size, in_features, bias=bias)

    def forward(self, x):
        x, gate = self.w1(x).chunk(2, dim=-1)
        x = self.w2(F.silu(x) * gate)
        return x