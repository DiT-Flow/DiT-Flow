import torch
import torch.nn as nn
from .stable_vae import load_vae
from .stft_vae import load_stft_vae


class Autoencoder(nn.Module):
    def __init__(self, ckpt_path, config_file, model_type='stable_vae', quantization_first=True):
        super(Autoencoder, self).__init__()
        self.model_type = model_type
        if self.model_type == 'stable_vae':
            model = load_vae(ckpt_path, config_file)
        elif self.model_type == 'stft_vae':
            model = load_stft_vae(ckpt_path, config_file)
        else:
            raise NotImplementedError(f"Model type not implemented: {self.model_type}")
        self.ae = model.eval()
        self.quantization_first = quantization_first
        print(f'Autoencoder quantization first mode: {quantization_first}')

    @torch.no_grad()
    def forward(self, audio=None, embedding=None, std=None):
        if self.model_type == 'stable_vae':
            return self.process_stable_vae(audio, embedding)
        elif self.model_type == 'stft_vae':
            return self.process_stft_vae(audio, embedding, std)
        else:
            raise NotImplementedError(f"Model type not implemented: {self.model_type}")

    def process_stable_vae(self, audio=None, embedding=None):
        if audio is not None:
            z = self.ae.encoder(audio)
            if self.quantization_first:
                z = self.ae.bottleneck.encode(z)
            return z
        if embedding is not None:
            z = embedding
            if self.quantization_first:
                audio = self.ae.decoder(z)
            else:
                z = self.ae.bottleneck.encode(z)
                audio = self.ae.decoder(z)
            return audio
        else:
            raise ValueError("Either audio or embedding must be provided.")

    def process_stft_vae(self, audio=None, embedding=None, std=None):
        if audio is not None:
            z, std = self.ae.encoder(audio)
            if self.quantization_first:
                z = self.ae.bottleneck.encode(z)
            return z, std
        if embedding is not None:
            z = embedding
            if self.quantization_first:
                audio = self.ae.decoder(z, std)
            else:
                z = self.ae.bottleneck.encode(z)
                audio = self.ae.decoder(z, std)
            return audio
        else:
            raise ValueError("Either audio or embedding must be provided.")
