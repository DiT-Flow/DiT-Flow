import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import repeat
import math
from .udit import UDiT
from .uvit import UViT
from .utils.span_mask import compute_mask_indices
from .vit import ViT


def create_mask(x, lengths):
    """
    Create a boolean mask for sequences of varying lengths.

    Args:
        x (torch.Tensor): The input tensor of shape (B, L).
        lengths (torch.Tensor): A tensor of lengths for each sequence in the batch, shape (B,).

    Returns:
        torch.Tensor: A boolean mask tensor of shape (B, L).
    """
    device = x.device  # Ensure the mask is created on the same device as `x`
    B, L, _ = x.shape
    lengths = lengths.to(device)  # Ensure lengths are on the same device
    mask = torch.arange(L, device=device).unsqueeze(0) < lengths.unsqueeze(1)
    return mask.bool()


class SoloSpeech_TSE(nn.Module):
    def __init__(self,
                 udit_model=None, 
                 vit_model=None,
                 ):
        super().__init__()
        self.model = UDiT(**udit_model)
        out_channel = udit_model.pop('out_chans', None)
        context_dim = udit_model.pop('context_dim', None)

        self.spk_proj = nn.Linear(out_channel, context_dim)
        self.spk_model = ViT(**vit_model)

    def forward(self, x, timesteps, mixture, reference,
                x_len=None, ref_len=None):

        reference = reference.clone()

        x_mask = create_mask(x, x_len)
        ref_mask = create_mask(reference, ref_len)

        ref = self.spk_proj(reference)
        ref = self.spk_model(x=ref, x_mask=ref_mask)

        x = torch.cat([x, mixture], dim=-1)
        x = x.transpose(1, 2)
        x = self.model(x=x, timesteps=timesteps, context=ref,
                       x_mask=x_mask, context_mask=ref_mask,
                       cls_token=None)
        # this cls token can handle global embeddings
        x = x.transpose(1, 2)
        return x, x_mask

class SoloSpeech_Disc(nn.Module):
    def __init__(self,
                 uvit_model=None, 
                 vit_model=None,
                 ):
        super().__init__()
        self.model = UViT(**uvit_model)
        out_channel = uvit_model.pop('out_chans', None)
        context_dim = uvit_model.pop('context_dim', None)

        self.spk_proj = nn.Linear(out_channel, context_dim)
        self.spk_model = ViT(**vit_model)

    def forward(self, x, reference,
                x_len=None, ref_len=None):

        reference = reference.clone()

        x_mask = create_mask(x, x_len)
        ref_mask = create_mask(reference, ref_len)

        ref = self.spk_proj(reference)
        ref = self.spk_model(x=ref, x_mask=ref_mask)

        x = x.transpose(1, 2)
        x = self.model(x=x, context=ref,
                       x_mask=x_mask, context_mask=ref_mask,
                       cls_token=None)
        # this cls token can handle global embeddings
        x = x.transpose(1, 2)
        return x, x_mask

class SoloSpeech_TSR(nn.Module):
    def __init__(self,
                 udit_model=None, 
                 ):
        super().__init__()
        self.model = UDiT(**udit_model)

    def forward(self, x, timesteps, mixture,
                x_len=None):

        x_mask = create_mask(x, x_len)
        x = torch.cat([x, mixture], dim=-1)
        x = x.transpose(1, 2)
        x = self.model(x=x, timesteps=timesteps, context=None,
                       x_mask=x_mask, context_mask=None,
                       cls_token=None)
        # this cls token can handle global embeddings
        x = x.transpose(1, 2)
        return x, x_mask