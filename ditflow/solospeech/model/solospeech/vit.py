import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint
from .utils.attention import Attention
from .utils.modules import FeedForward
from .utils.modules import PE_wrapper


class ViTBlock(nn.Module):
    def __init__(self, dim, context_dim=None,
                 num_heads=8, mlp_ratio=4.,
                 qkv_bias=False, qk_scale=None, qk_norm='rmsnorm',
                 norm_layer='rmsnorm',
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

        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim=dim,
                              num_heads=num_heads,
                              qkv_bias=qkv_bias, qk_scale=qk_scale,
                              qk_norm=qk_norm,
                              rope_mode=rope_mode)

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

        self.norm3 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = FeedForward(in_features=dim,
                               hidden_size=mlp_hidden_dim,
                               multiple_of=256, bias=mlp_bias)

        self.use_checkpoint = use_checkpoint

    def forward(self, x, context=None,
                x_mask=None, context_mask=None, extras=None):
        if self.use_checkpoint:
            return checkpoint(self._forward, x,
                              context,
                              x_mask, context_mask, extras,
                              use_reentrant=False)
        else:
            return self._forward(x, context,
                                 x_mask, context_mask, extras)

    def _forward(self, x, context=None,
                 x_mask=None, context_mask=None, extras=None):
        # self attention
        x_norm = self.norm1(x)
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
        x_norm = self.norm3(x)
        x_mlp = self.mlp(x_norm)
        x = x + x_mlp

        return x


class ViT(nn.Module):
    def __init__(self,
                 in_chans=128, out_chans=None, seq_len=128,
                 embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4.,
                 qkv_bias=False, qk_scale=None, qk_norm='layernorm',
                 norm_layer='layernorm',
                 context_norm=False,
                 use_checkpoint=False,
                 context_dim=None, context_fusion=None,
                 context_max_length=None, context_pe_method='none',
                 pe_method='none', rope_mode='shared',):
        super().__init__()
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models

        # input
        self.in_chans = in_chans
        self.in_proj = nn.Linear(in_chans, embed_dim)
        out_chans = in_chans if out_chans is None else out_chans
        self.out_chans = out_chans

        # position embedding
        self.rope = rope_mode
        self.x_pe = PE_wrapper(dim=embed_dim, method=pe_method,
                               length=seq_len)

        print(f'x position embedding: {pe_method}')
        print(f'rope mode: {self.rope}')

        self.extras = 0
        # context
        self.use_context = False
        self.context_cross = False
        self.context_max_length = context_max_length
        self.context_fusion = 'none'
        if context_dim is not None:
            self.use_context = True
            self.context_embed = nn.Sequential(
                nn.Linear(context_dim, embed_dim, bias=True),
                nn.SiLU(),
                nn.Linear(embed_dim, embed_dim, bias=True),)
            self.context_fusion = context_fusion
            self.context_pe = PE_wrapper(dim=embed_dim,
                                         method=context_pe_method,
                                         length=context_max_length)
            if context_fusion == 'concat':
                self.extras += context_max_length
                # no cross attention layers
                context_dim = None
            elif context_fusion == 'cross':
                self.context_cross = True
                context_dim = embed_dim
            else:
                raise NotImplementedError
        print(f'context fusion mode: {context_fusion}')
        print(f'context position embedding: {context_pe_method}')

        block_list = []
        for i in range(depth):
            block_list.append(ViTBlock(
                dim=embed_dim, context_dim=context_dim, num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias, qk_scale=qk_scale, qk_norm=qk_norm,
                norm_layer=norm_layer,
                rope_mode=self.rope,
                context_norm=context_norm,
                use_checkpoint=use_checkpoint))
        self.blocks = nn.ModuleList(block_list)

        # FinalLayer block
        self.norm = nn.LayerNorm(embed_dim)
        self.final_layer = nn.Linear(embed_dim, out_chans)

    def _concat_x_context(self, x, context, x_mask=None, context_mask=None):
        assert context.shape[-2] == self.context_max_length
        # Check if either x_mask or context_mask is provided
        B = x.shape[0]
        # Create default masks if they are not provided
        if x_mask is None:
            x_mask = torch.ones(B, x.shape[-2], device=x.device).bool()
        if context_mask is None:
            context_mask = torch.ones(B, context.shape[-2],
                                      device=context.device).bool()
        # Concatenate the masks along the second dimension (dim=1)
        x_mask = torch.cat([context_mask, x_mask], dim=1).bool()
        # Concatenate context and x along the second dimension (dim=1)
        x = torch.cat([context, x], dim=1)
        return x, x_mask

    def forward(self, x, x_mask=None, context=None, context_mask=None):
        x = self.in_proj(x)
        x = self.x_pe(x)

        if self.use_context:
            context_token = self.context_embed(context)
            context_token = self.context_pe(context_token)
            if self.context_fusion == 'concat':
                x, x_mask = self._concat_x_context(x=x, context=context_token,
                                                   x_mask=x_mask,
                                                   context_mask=context_mask)
                context_token, context_mask = None, None
        else:
            context_token, context_mask = None, None

        for block in self.blocks:
            x = block(x, context=context, x_mask=x_mask, context_mask=context_mask)

        x = self.norm(x)
        x = self.final_layer(x)

        return x

'''
example
if __name__ == '__main__':
    x = torch.rand(4, 128, 128)

    model = ViT(in_chans=128, out_chans=None, seq_len=128,
                embed_dim=768, depth=12,
                num_heads=12, mlp_ratio=4.,
                qkv_bias=False, qk_scale=None, qk_norm='layernorm',
                norm_layer='layernorm',
                context_norm=False,
                use_checkpoint=True,
                context_dim=None, context_fusion=None,
                context_max_length=None, context_pe_method='none',
                pe_method='none', rope_mode='shared',)

    y = model(x)

    total_params = sum([param.nelement() for param in model.parameters()])
    print("Number of parameter: %.2fM" % (total_params / 1e6))
'''