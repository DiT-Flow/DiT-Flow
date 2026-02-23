import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint
from .utils.attention import Attention, JointAttention
from .utils.modules import unpatchify, FeedForward
from .modnorm import ModulatedRMSNorm, ModulatedLayerNorm


class UViTBlock(nn.Module):
    """
    A modified PixArt block with adaptive layer norm (adaLN-single) conditioning.
    """

    def __init__(self, dim, context_dim=None,
                 num_heads=8, mlp_ratio=4.,
                 qkv_bias=False, qk_scale=None, qk_norm='rmsnorm',
                 norm_layer='rmsnorm',
                 skip=False, skip_norm=False,
                 rope_mode='none',
                 context_norm=True,
                 use_checkpoint=False):

        super().__init__()
        if norm_layer == 'rmsnorm':
            norm_layer = nn.RMSNorm
            mlp_bias = False
        elif norm_layer == 'layernorm':
            norm_layer = nn.LayerNorm
            mlp_bias = True
        else:
            NotImplementedError

        self.mod_norm1 = norm_layer(dim)
        self.attn = Attention(dim=dim,
                              num_heads=num_heads,
                              qkv_bias=qkv_bias, qk_scale=qk_scale,
                              qk_norm=qk_norm,
                              rope_mode=rope_mode)
        # self.gate_norm1 = ResidualTanhGatedRMSNorm(dim)

        if context_dim is not None:
            self.use_context = True
            self.cross_attn = Attention(dim=dim,
                                        num_heads=num_heads,
                                        context_dim=context_dim,
                                        qkv_bias=qkv_bias, qk_scale=qk_scale,
                                        qk_norm=qk_norm,
                                        rope_mode='none')
            self.norm2 = norm_layer(dim)
            if context_norm:
                self.norm_context = norm_layer(context_dim)
            else:
                self.norm_context = nn.Identity()
        else:
            self.use_context = False

        self.mod_norm3 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = FeedForward(in_features=dim,
                               hidden_size=mlp_hidden_dim,
                               multiple_of=256, bias=mlp_bias)

        if skip:
            self.skip_norm = norm_layer(2 * dim) if skip_norm else nn.Identity()
            self.skip_linear = nn.Linear(2 * dim, dim)
        else:
            self.skip_linear = None

        self.use_checkpoint = use_checkpoint

    def forward(self, x,
                skip=None, context=None,
                x_mask=None, context_mask=None, extras=None):
        if self.use_checkpoint:
            return checkpoint(self._forward, x,
                              skip, context,
                              x_mask, context_mask, extras,
                              use_reentrant=False)
        else:
            return self._forward(x,
                                 skip, context,
                                 x_mask, context_mask, extras)

    def _forward(self, x,
                 skip=None, context=None,
                 x_mask=None, context_mask=None, extras=None):
        B, T, C = x.shape
        if self.skip_linear is not None:
            assert skip is not None
            cat = torch.cat([x, skip], dim=-1)
            cat = self.skip_norm(cat)
            x = self.skip_linear(cat)

        # self attention
        x_norm = self.mod_norm1(x)
        x_attn = self.attn(x_norm, context=None,
                           context_mask=x_mask,
                           extras=extras)
        x = x + x_attn

        # cross attention
        if self.use_context:
            assert context is not None
            x = x + self.cross_attn(x=self.norm2(x),
                                    context=self.norm_context(context),
                                    context_mask=context_mask, extras=extras)

        # mlp
        x_norm = self.mod_norm3(x)
        x_mlp = self.mlp(x_norm)
        x = x + x_mlp

        return x


class FinalBlock(nn.Module):
    def __init__(self, embed_dim, patch_size, in_chans,
                 img_size,
                 input_type='2d',
                 use_conv=False):
        super().__init__()
        self.in_chans = in_chans
        self.img_size = img_size
        self.input_type = input_type

        self.norm = nn.LayerNorm(embed_dim)

        if input_type == '2d':
            self.patch_dim = patch_size ** 2 * in_chans
            self.linear = nn.Linear(embed_dim, self.patch_dim, bias=True)
            if use_conv:
                self.final_layer = nn.Conv2d(self.in_chans, self.in_chans, 
                                             3, padding=1)
            else:
                self.final_layer = nn.Identity()

        elif input_type == '1d':
            self.patch_dim = patch_size * in_chans
            self.linear = nn.Linear(embed_dim, self.patch_dim, bias=True)
            if use_conv:
                self.final_layer = nn.Conv1d(self.in_chans, self.in_chans, 
                                             3, padding=1)
            else:
                self.final_layer = nn.Identity()

    def forward(self, x, extras=0):
        B, T, C = x.shape
        x = x[:, extras:, :]
        # only handle generation target
        x = self.norm(x)
        x = self.linear(x)
        x = unpatchify(x, self.in_chans, self.input_type, self.img_size)
        x = self.final_layer(x)
        return x
