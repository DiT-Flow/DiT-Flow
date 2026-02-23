import torch
import torch.nn as nn
import torch.utils.checkpoint
from .utils.modules import PatchEmbed, TimestepEmbedder
from .utils.modules import PE_wrapper
from .v_blocks import UViTBlock, FinalBlock


class UViT(nn.Module):
    def __init__(self,
                 img_size=500, patch_size=1, in_chans=128,
                 input_type='1d', out_chans=None,
                 embed_dim=768, depth=12, joint_block=0,
                 num_heads=12, mlp_ratio=4.,
                 qkv_bias=False, qk_scale=None, qk_norm='layernorm',
                 norm_layer='layernorm',
                 context_norm=True,
                 use_checkpoint=True,
                 # time fusion ada or token
                 cls_dim=None,
                 # max length is only used for concat
                 context_dim=384, context_fusion='cross', context_max_length=None, context_pe_method='none',
                 pe_method='none', rope_mode='shared',
                 use_conv=False,
                 skip=True, skip_norm=True):
        super().__init__()
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models

        # input
        self.in_chans = in_chans
        self.input_type = input_type
        if self.input_type == '2d':
            num_patches = (img_size[0] // patch_size) * (img_size[1] // patch_size)
        elif self.input_type == '1d':
            num_patches = img_size // patch_size
        self.patch_embed = PatchEmbed(patch_size=patch_size, in_chans=in_chans,
                                      embed_dim=embed_dim, input_type=input_type)
        out_chans = in_chans if out_chans is None else out_chans
        self.out_chans = out_chans

        # position embedding
        self.rope = rope_mode
        self.x_pe = PE_wrapper(dim=embed_dim, method=pe_method,
                               length=num_patches)

        print(f'x position embedding: {pe_method}')
        print(f'rope mode: {self.rope}')

        # cls embed
        if cls_dim is not None:
            self.cls_embed = nn.Sequential(
                nn.Linear(cls_dim, embed_dim, bias=True),
                nn.SiLU(),
                nn.Linear(embed_dim, embed_dim, bias=True),)
        else:
            self.cls_embed = None

        # time fusion
        self.extras = 0

        # context
        # use a simple projection
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
            elif context_fusion == 'e2':
                context_dim = None
                self.e2fusion = nn.Linear(embed_dim*2, embed_dim)
            else:
                raise NotImplementedError
        print(f'context fusion mode: {context_fusion}')
        print(f'context position embedding: {context_pe_method}')

        self.use_skip = skip
        assert joint_block <= depth // 2
        self.joint_block = joint_block

        print(f'use long skip connection: {skip}')
        in_block_list = []
        for _ in range(depth // 2):
            if joint_block > 0:
                Block = UViTBlock
                joint_block = joint_block - 1
            else:
                Block = UViTBlock
            in_block_list.append(Block(
                dim=embed_dim, context_dim=context_dim, num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias, qk_scale=qk_scale, qk_norm=qk_norm,
                norm_layer=norm_layer,
                skip=False, skip_norm=False,
                rope_mode=self.rope,
                context_norm=context_norm,
                use_checkpoint=use_checkpoint))
        self.in_blocks = nn.ModuleList(in_block_list)

        self.mid_block = UViTBlock(
            dim=embed_dim, context_dim=context_dim, num_heads=num_heads, 
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias, qk_scale=qk_scale, qk_norm=qk_norm,
            norm_layer=norm_layer,
            skip=False, skip_norm=False,
            rope_mode=self.rope,
            context_norm=context_norm,
            use_checkpoint=use_checkpoint)

        out_block_list = []
        for _ in range(depth // 2):
            out_block_list.append(
                UViTBlock(
                    dim=embed_dim, context_dim=context_dim, num_heads=num_heads, 
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias, qk_scale=qk_scale, qk_norm=qk_norm,
                    norm_layer=norm_layer,
                    skip=skip, skip_norm=skip_norm,
                    rope_mode=self.rope,
                    context_norm=context_norm,
                    use_checkpoint=use_checkpoint))
        self.out_blocks = nn.ModuleList(out_block_list)

        # FinalLayer block
        self.use_conv = use_conv
        self.final_block = FinalBlock(embed_dim=embed_dim,
                                      patch_size=patch_size,
                                      img_size=img_size,
                                      in_chans=out_chans,
                                      input_type=input_type,
                                      use_conv=use_conv)
        self.initialize_weights()

    def initialize_weights(self):
        # Basic init for all layers
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

        # init patch Conv like Linear
        w = self.patch_embed.proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        nn.init.constant_(self.patch_embed.proj.bias, 0)

        # Zero-out Cross Attention
        if self.context_cross:
            for block in self.in_blocks:
                nn.init.constant_(block.cross_attn.proj.weight, 0)
                nn.init.constant_(block.cross_attn.proj.bias, 0)
            nn.init.constant_(self.mid_block.cross_attn.proj.weight, 0)
            nn.init.constant_(self.mid_block.cross_attn.proj.bias, 0)
            for block in self.out_blocks:
                nn.init.constant_(block.cross_attn.proj.weight, 0)
                nn.init.constant_(block.cross_attn.proj.bias, 0)

        # init out Conv
        if self.use_conv:
            nn.init.xavier_uniform_(self.final_block.final_layer.weight)
            nn.init.constant_(self.final_block.final_layer.bias, 0)

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


    def forward(self, x, context, 
                x_mask=None, context_mask=None,
                cls_token=None,):

        x = self.patch_embed(x)
        x = self.x_pe(x)

        B, L, D = x.shape

        if self.use_context:
            context_token = self.context_embed(context)
            context_token = self.context_pe(context_token)
            if self.context_fusion == 'concat':
                x, x_mask = self._concat_x_context(x=x, context=context_token,
                                                   x_mask=x_mask,
                                                   context_mask=context_mask)
                context_token, context_mask = None, None
            elif self.context_fusion == 'e2':
                x = torch.cat([x, context_token], dim=-1)
                x = self.e2fusion(x)
                context_token, context_mask = None, None
        else:
            context_token, context_mask = None, None

        skips = []

        for blk in self.in_blocks:
            x = blk(x=x,
                    skip=None, context=context_token,
                    x_mask=x_mask, context_mask=context_mask,
                    extras=self.extras)
            if self.use_skip:
                skips.append(x)

        x = self.mid_block(x=x,
                           skip=None, context=context_token,
                           x_mask=x_mask, context_mask=context_mask,
                           extras=self.extras)

        for blk in self.out_blocks:
            skip = skips.pop() if self.use_skip else None
            x = blk(x=x,
                    skip=skip, context=context_token,
                    x_mask=x_mask, context_mask=context_mask,
                    extras=self.extras)

        x = self.final_block(x, extras=self.extras)

        return x