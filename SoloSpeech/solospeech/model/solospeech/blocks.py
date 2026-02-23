import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint
from .utils.attention import Attention, JointAttention
from .utils.modules import unpatchify, FeedForward
from .utils.modules import film_modulate
from .modnorm import ModulatedRMSNorm, ModulatedLayerNorm


class DiTBlock(nn.Module):
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
            ModNorm = ModulatedRMSNorm
            norm_layer = nn.RMSNorm
            self.ada_out = 4
            mlp_bias = False
        elif norm_layer == 'layernorm':
            ModNorm = ModulatedLayerNorm
            norm_layer = nn.LayerNorm
            self.ada_out = 6
            mlp_bias = True
        else:
            NotImplementedError

        self.mod_norm1 = ModNorm(dim)
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

        self.mod_norm3 = ModNorm(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = FeedForward(in_features=dim,
                               hidden_size=mlp_hidden_dim,
                               multiple_of=256, bias=mlp_bias)
        # self.gate_norm3 = ResidualTanhGatedRMSNorm(dim)

        self.adaln = nn.Linear(dim, self.ada_out * dim, bias=True)

        if skip:
            self.skip_norm = norm_layer(2 * dim) if skip_norm else nn.Identity()
            self.skip_linear = nn.Linear(2 * dim, dim)
        else:
            self.skip_linear = None

        self.use_checkpoint = use_checkpoint

    def forward(self, x, time_token=None,
                skip=None, context=None,
                x_mask=None, context_mask=None, extras=None):
        if self.use_checkpoint:
            return checkpoint(self._forward, x,
                              time_token, skip, context,
                              x_mask, context_mask, extras,
                              use_reentrant=False)
        else:
            return self._forward(x,
                                 time_token, skip, context,
                                 x_mask, context_mask, extras)

    def _forward(self, x, time_token=None,
                 skip=None, context=None,
                 x_mask=None, context_mask=None, extras=None):
        B, T, C = x.shape
        if self.skip_linear is not None:
            assert skip is not None
            cat = torch.cat([x, skip], dim=-1)
            cat = self.skip_norm(cat)
            x = self.skip_linear(cat)

        time_ada = self.adaln(time_token)

        if self.ada_out == 4:
            (scale_msa, gate_msa,
             scale_mlp, gate_mlp) = time_ada.reshape(B, 4, -1).chunk(4, dim=1)
            shift_msa, shift_mlp = None, None
        elif self.ada_out == 6:
            (scale_msa, gate_msa, shift_msa,
             scale_mlp, gate_mlp, shift_mlp) = time_ada.reshape(B, 6, -1).chunk(6, dim=1)

        # self attention
        x_norm = self.mod_norm1(x, scale_msa, shift_msa)
        x_attn = self.attn(x_norm, context=None,
                           context_mask=x_mask,
                           extras=extras)
        tanh_gate_msa = torch.tanh(1 - gate_msa)
        x = x + x_attn * tanh_gate_msa

        # cross attention
        if self.use_context:
            assert context is not None
            x = x + self.cross_attn(x=self.norm2(x),
                                    context=self.norm_context(context),
                                    context_mask=context_mask, extras=extras)

        # mlp
        x_norm = self.mod_norm3(x, scale_mlp, shift_mlp)
        x_mlp = self.mlp(x_norm)
        tanh_gate_mlp = torch.tanh(1 - gate_mlp)
        x = x + x_mlp * tanh_gate_mlp

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

        self.norm = nn.LayerNorm(embed_dim, elementwise_affine=False)

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

    def forward(self, x, time_ada=None, extras=0):
        B, T, C = x.shape
        x = x[:, extras:, :]
        # only handle generation target
        shift, scale = time_ada.reshape(B, 2, -1).chunk(2, dim=1)
        x = film_modulate(self.norm(x), shift, scale)
        x = self.linear(x)
        x = unpatchify(x, self.in_chans, self.input_type, self.img_size)
        x = self.final_layer(x)
        return x


# class JointDiTBlock(nn.Module):
#     """
#     A modified PixArt block with adaptive layer norm (adaLN-single) conditioning.
#     """

#     def __init__(self, dim, context_dim=None,
#                  num_heads=8, mlp_ratio=4.,
#                  qkv_bias=False, qk_scale=None, qk_norm=None,
#                  act_layer='gelu', norm_layer=nn.LayerNorm,
#                  time_fusion='none',
#                  ada_lora_rank=None, ada_lora_alpha=None,
#                  skip=False, skip_norm=False,
#                  rope_mode=False,
#                  context_norm=False,
#                  use_checkpoint=False,):

#         super().__init__()
#         # no cross attention
#         assert context_dim is None
#         self.attn_norm_x = norm_layer(dim)
#         self.attn_norm_c = norm_layer(dim)
#         self.attn = JointAttention(dim=dim,
#                                    num_heads=num_heads,
#                                    qkv_bias=qkv_bias, qk_scale=qk_scale,
#                                    qk_norm=qk_norm,
#                                    rope_mode=rope_mode)
#         self.ffn_norm_x = norm_layer(dim)
#         self.ffn_norm_c = norm_layer(dim)
#         self.mlp_x = FeedForward(dim=dim, mult=mlp_ratio,
#                                  activation_fn=act_layer, dropout=0)
#         self.mlp_c = FeedForward(dim=dim, mult=mlp_ratio,
#                                  activation_fn=act_layer, dropout=0)

#         # Zero-out the shift table
#         self.use_adanorm = True if time_fusion != 'token' else False
#         if self.use_adanorm:
#             self.adaln_x = AdaLN(dim, ada_mode=time_fusion,
#                                  r=ada_lora_rank, alpha=ada_lora_alpha)
#             self.adaln_c = AdaLN(dim, ada_mode=time_fusion,
#                                  r=ada_lora_rank, alpha=ada_lora_alpha)

#         # if skip is False:
#         #     skip_x, skip_c = False, False
#         # else:
#         #     skip_x, skip_c = skip, skip

#         self.skip_linear_x = nn.Linear(2 * dim, dim) if skip else None
#         self.skip_linear_c = nn.Linear(2 * dim, dim) if skip else None
#         self.skip_norm_x = norm_layer(2 * dim) if skip_norm else nn.Identity()
#         self.skip_norm_c = norm_layer(2 * dim) if skip_norm else nn.Identity()

#         self.use_checkpoint = use_checkpoint

#     def forward(self, x, time_token=None, time_ada=None,
#                 skip=None, context=None,
#                 x_mask=None, context_mask=None, extras=None):
#         if self.use_checkpoint:
#             return checkpoint(self._forward, x,
#                               time_token, time_ada, skip,
#                               context, x_mask, context_mask, extras,
#                               use_reentrant=False)
#         else:
#             return self._forward(x,
#                                  time_token, time_ada, skip,
#                                  context, x_mask, context_mask, extras)

#     def _forward(self, x, time_token=None, time_ada=None,
#                  skip=None, context=None,
#                  x_mask=None, context_mask=None, extras=None):

#         assert context is None and context_mask is None

#         context, x = x[:, :extras, :], x[:, extras:, :]
#         context_mask, x_mask = x_mask[:, :extras], x_mask[:, extras:]

#         if skip is not None:
#             skip_c, skip_x = skip[:, :extras, :], skip[:, extras:, :]

#         # B, T, C = x.shape
#         if self.skip_linear_x is not None:
#             x = self.skip_linear_x(self.skip_norm_x(torch.cat([x, skip_x]), dim=-1))

#         if self.skip_linear_c is not None:
#             context = self.skip_linear_c(self.skip_norm_c(torch.cat([context, skip_c]), dim=-1))

#         if self.use_adanorm:
#             time_ada_x = self.adaln_x(time_token, time_ada)
#             (shift_msa_x, scale_msa_x, gate_msa_x,
#              shift_mlp_x, scale_mlp_x, gate_mlp_x) = time_ada_x.chunk(6, dim=1)

#             time_ada_c = self.adaln_c(time_token, time_ada)
#             (shift_msa_c, scale_msa_c, gate_msa_c,
#              shift_mlp_c, scale_mlp_c, gate_mlp_c) = time_ada_c.chunk(6, dim=1)

#         # self attention
#         x_norm = self.attn_norm_x(x)
#         c_norm = self.attn_norm_c(context)
#         if self.use_adanorm:
#             x_norm = film_modulate(x_norm, shift=shift_msa_x, scale=scale_msa_x)
#             c_norm = film_modulate(c_norm, shift=shift_msa_c, scale=scale_msa_c)

#         x_out, c_out = self.attn(x_norm, context=c_norm,
#                                  x_mask=x_mask, context_mask=context_mask,
#                                  extras=extras)
#         if self.use_adanorm:
#             x = x + (1 - gate_msa_x) * x_out
#             context = context + (1 - gate_msa_c) * c_out
#         else:
#             x = x + x_out
#             context = context + c_out

#         # mlp
#         x_norm = self.ffn_norm_x(x)
#         c_norm = self.ffn_norm_c(context)
#         if self.use_adanorm:
#             x_norm = film_modulate(x_norm, shift=shift_mlp_x, scale=scale_mlp_x)
#             x = x + (1 - gate_mlp_x) * self.mlp_x(x_norm)

#             c_norm = film_modulate(c_norm, shift=shift_mlp_c, scale=scale_mlp_c)
#             context = context + (1 - gate_mlp_c) * self.mlp_c(c_norm)
#         else:
#             x = x + self.mlp_x(x_norm)
#             context = context + self.mlp_c(c_norm)

#         x = torch.cat([context, x], dim=1)
#         return x