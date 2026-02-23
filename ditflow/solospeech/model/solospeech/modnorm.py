import torch
import torch.nn as nn


class ModulatedRMSNorm(nn.Module):
    def __init__(self, dim):
        super(ModulatedRMSNorm, self).__init__()
        self.norm = nn.RMSNorm(dim, elementwise_affine=False)

    def forward(self, x, scale, shift=None):
        x_normed = self.norm(x)
        x_modulated = x_normed * (1 + scale)
        return x_modulated


class ModulatedLayerNorm(nn.Module):
    def __init__(self, dim):
        super(ModulatedLayerNorm, self).__init__()
        self.norm = nn.LayerNorm(dim, elementwise_affine=False)

    def forward(self, x, scale, shift):
        # Apply LayerNorm
        x_normed = self.norm(x)
        # Modulate with scale
        x_modulated = x_normed * (1 + scale) + shift
        return x_modulated


class ResidualTanhGatedRMSNorm(nn.Module):
    def __init__(self, dim):
        super(ResidualTanhGatedRMSNorm, self).__init__()
        self.norm = nn.RMSNorm(dim)

    def forward(self, x, x_res, gate):
        x_res = self.norm(x_res)

        tanh_gate = torch.tanh(1 - gate)
        output = x + x_res * tanh_gate

        return output


class ResidualTanhGatedLayerNorm(nn.Module):
    def __init__(self, dim):
        super(ResidualTanhGatedLayerNorm, self).__init__()
        self.norm = nn.LayerNorm(dim)

    def forward(self, x, x_res, gate):
        # Apply LayerNorm on the residual input
        x_res = self.norm(x_res)

        tanh_gate = torch.tanh(1 - gate)
        output = x + x_res * tanh_gate
        return output