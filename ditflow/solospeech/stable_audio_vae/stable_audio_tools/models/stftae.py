import torch
import math
import numpy as np

from torch import nn
from torch.nn import functional as F
from torchaudio import transforms as T
from typing import Literal, Dict, Any
import copy
from ..inference.sampling import sample
from ..inference.utils import prepare_audio
from .bottleneck import Bottleneck, DiscreteBottleneck
from .factory import create_pretransform_from_config, create_bottleneck_from_config
from .pretransforms import Pretransform
from .TFgridnet import GridNetV2Block
from .feature import STFT, iSTFT

def checkpoint(function, *args, **kwargs):
    kwargs.setdefault("use_reentrant", False)
    return torch.utils.checkpoint.checkpoint(function, *args, **kwargs)

class OobleckEncoder(nn.Module):
    def __init__(
        self, 
        in_channels=1,
        latent_dim=32, 
        n_fft=640,
        hop_length=320,
        win_length=640,
        hidden_channels=256,
        n_head=4,
        approx_qk_dim=512,
        emb_dim=128,
        emb_ks=1,
        emb_hs=1,
        num_layers=6,
        eps=1e-6,
        ):
        super().__init__()
        
        self.num_layers = num_layers
        self.stft = STFT(n_fft=n_fft, hop_length=hop_length, win_length=win_length)
        n_freqs = n_fft//2 + 1
        t_ksize = 3
        ks, padding = (t_ksize, 3), (t_ksize // 2, 1)

        self.conv = nn.Sequential(
            nn.Conv2d(2, emb_dim, ks, padding=padding),
            nn.GroupNorm(1, emb_dim, eps=eps),
        )
        self.dual_mdl = nn.ModuleList([])
        for i in range(num_layers):
            self.dual_mdl.append(
                copy.deepcopy(
                    GridNetV2Block(
                        emb_dim,
                        emb_ks,
                        emb_hs,
                        n_freqs,
                        hidden_channels,
                        n_head,
                        approx_qk_dim=approx_qk_dim,
                        activation="prelu",
                    )
                )
            )

        self.proj = nn.Sequential(
            nn.PReLU(),
            nn.Conv1d(emb_dim * n_freqs, latent_dim, kernel_size=1)
        )
        self.eps = eps

    def forward(self, x):
        # print('1:\t', x.shape) # torch.Size([32, 1, 15680])
        assert x.ndim == 3, x.shape
        std = x.std(dim=(1, 2), keepdim=True)+self.eps
        x = x / std
        x = self.stft(x)[-1] # (B, N, T, C) 
        x = torch.cat([x.real, x.imag],dim = 1)
        # print('2:\t', x.shape) # torch.Size([32, 2, 321, 50])
        x = x.permute(0,1,3,2).contiguous() 
        # print('3:\t', x.shape) # torch.Size([32, 2, 50, 321])
        x = self.conv(x)
        # print('4:\t', x.shape) # torch.Size([32, 256, 50, 321])
        for i in range(self.num_layers):
            x = self.dual_mdl[i](x)
        # print('5:\t', x.shape) # torch.Size([32, 256, 50, 321])
        x = x.permute(0,1,3,2).contiguous() 
        bs, n_c, n_f, n_t = x.shape
        x = x.reshape(bs, -1, n_t)
        # print('6:\t', x.shape) # torch.Size([32, 41088, 50])
        x = self.proj(x)
        # print('7:\t', x.shape) # torch.Size([32, 256, 50])
        return x, std


class OobleckDecoder(nn.Module):
    def __init__(
        self, 
        out_channels=1,
        latent_dim=128, 
        n_fft=640,
        hop_length=320,
        win_length=640,
        hidden_channels=512,
        n_head=4,
        approx_qk_dim=512,
        emb_dim=256,
        emb_ks=1,
        emb_hs=1,
        num_layers=6,
        eps=1e-6,
        ):
        super().__init__()

        self.num_layers = num_layers
        self.istft = iSTFT(n_fft=n_fft, hop_length=hop_length, win_length=win_length)
        n_freqs = n_fft//2 + 1
        t_ksize = 3
        ks, padding = (t_ksize, 3), (t_ksize // 2, 1)

        self.deconv = nn.ConvTranspose2d(emb_dim, 2, ks, padding=padding)
        
        self.dual_mdl = nn.ModuleList([])
        for i in range(num_layers):
            self.dual_mdl.append(
                copy.deepcopy(
                    GridNetV2Block(
                        emb_dim,
                        emb_ks,
                        emb_hs,
                        n_freqs,
                        hidden_channels,
                        n_head,
                        approx_qk_dim=approx_qk_dim,
                        activation="prelu",
                    )
                )
            )
        self.proj = nn.Conv1d(latent_dim, emb_dim * n_freqs, kernel_size=7, padding=3)
        self.n_freqs = n_freqs

    def forward(self, x, std=None):
        # print('1:\t', x.shape) # 1:	 torch.Size([32, 128, 50])
        x = self.proj(x)
        # print('2:\t', x.shape) # 2:	 torch.Size([32, 41088, 50])
        bs, _, n_t = x.shape
        x = x.reshape(bs, -1, self.n_freqs, n_t)
        # print('3:\t', x.shape) # 3:	 torch.Size([32, 128, 321, 50])
        x = x.permute(0,1,3,2).contiguous() 
        # print('4:\t', x.shape) # 4:	 torch.Size([32, 128, 50, 321])
        for i in range(self.num_layers):
            x = self.dual_mdl[i](x)
        # print('5:\t', x.shape) # 5:	 torch.Size([32, 128, 50, 321])
        x = self.deconv(x)
        # print('6:\t', x.shape) # 6:	 torch.Size([32, 2, 50, 321])
        out_r = x[:,0,:,:].permute(0,2,1).contiguous()
        out_i = x[:,1,:,:].permute(0,2,1).contiguous()
        est_source = self.istft((out_r, out_i), input_type="real_imag").unsqueeze(1)
        if std is not None:
            est_source = est_source * std
        # print('7:\t', est_source.shape) # torch.Size([16, 1, 25344])
        return est_source

class AudioAutoencoder(nn.Module):
    def __init__(
        self,
        encoder,
        decoder,
        latent_dim,
        downsampling_ratio,
        sample_rate,
        io_channels=2,
        bottleneck: Bottleneck = None,
        pretransform: Pretransform = None,
        in_channels = None,
        out_channels = None,
        soft_clip = False
    ):
        super().__init__()

        self.downsampling_ratio = downsampling_ratio
        self.sample_rate = sample_rate

        self.latent_dim = latent_dim
        self.io_channels = io_channels
        self.in_channels = io_channels
        self.out_channels = io_channels

        self.min_length = self.downsampling_ratio

        if in_channels is not None:
            self.in_channels = in_channels

        if out_channels is not None:
            self.out_channels = out_channels

        self.bottleneck = bottleneck

        self.encoder = encoder

        self.decoder = decoder

        self.pretransform = pretransform

        self.soft_clip = soft_clip
 
        self.is_discrete = self.bottleneck is not None and self.bottleneck.is_discrete

    def encode(self, audio, return_info=False, skip_pretransform=False, iterate_batch=False, **kwargs):

        info = {}

        if self.pretransform is not None and not skip_pretransform:
            if self.pretransform.enable_grad:
                if iterate_batch:
                    audios = []
                    for i in range(audio.shape[0]):
                        audios.append(self.pretransform.encode(audio[i:i+1]))
                    audio = torch.cat(audios, dim=0)
                else:
                    audio = self.pretransform.encode(audio)
            else:
                with torch.no_grad():
                    if iterate_batch:
                        audios = []
                        for i in range(audio.shape[0]):
                            audios.append(self.pretransform.encode(audio[i:i+1]))
                        audio = torch.cat(audios, dim=0)
                    else:
                        audio = self.pretransform.encode(audio)

        if self.encoder is not None:
            if iterate_batch:
                latents = []
                stds = []
                for i in range(audio.shape[0]):
                    tmp_latents, tmp_stds = self.encoder(audio[i:i+1])
                    latents.append(tmp_latents)
                    stds.append(tmp_stds)
                latents = torch.cat(latents, dim=0)
                stds = torch.cat(stds, dim=0)
            else:
                latents, stds = self.encoder(audio)
        else:
            latents = audio
            stds = None
        if self.bottleneck is not None:
            # TODO: Add iterate batch logic, needs to merge the info dicts
            latents, bottleneck_info = self.bottleneck.encode(latents, return_info=True, **kwargs)

            info.update(bottleneck_info)
        if return_info:
            return latents, stds, info

        return latents, stds

    def decode(self, latents, stds=None, iterate_batch=False, **kwargs):

        if self.bottleneck is not None:
            if iterate_batch:
                decoded = []
                for i in range(latents.shape[0]):
                    if stds is None:
                        decoded.append(self.bottleneck.decode(latents[i:i+1]))
                    else:
                        decoded.append(self.bottleneck.decode(latents[i:i+1]))
                decoded = torch.cat(decoded, dim=0)
            else:
                latents = self.bottleneck.decode(latents)

        if iterate_batch:
            decoded = []
            for i in range(latents.shape[0]):
                if stds is None:
                    decoded.append(self.decoder(latents[i:i+1]))
                else:
                    decoded.append(self.decoder(latents[i:i+1], stds[i:i+1]))
            decoded = torch.cat(decoded, dim=0)
        else:
            decoded = self.decoder(latents, stds, **kwargs)

        if self.pretransform is not None:
            if self.pretransform.enable_grad:
                if iterate_batch:
                    decodeds = []
                    for i in range(decoded.shape[0]):
                        decodeds.append(self.pretransform.decode(decoded[i:i+1]))
                    decoded = torch.cat(decodeds, dim=0)
                else:
                    decoded = self.pretransform.decode(decoded)
            else:
                with torch.no_grad():
                    if iterate_batch:
                        decodeds = []
                        for i in range(latents.shape[0]):
                            decodeds.append(self.pretransform.decode(decoded[i:i+1]))
                        decoded = torch.cat(decodeds, dim=0)
                    else:
                        decoded = self.pretransform.decode(decoded)

        if self.soft_clip:
            decoded = torch.tanh(decoded)
        
        return decoded
          
    def decode_tokens(self, tokens, **kwargs):
        '''
        Decode discrete tokens to audio
        Only works with discrete autoencoders
        '''

        assert isinstance(self.bottleneck, DiscreteBottleneck), "decode_tokens only works with discrete autoencoders"

        latents = self.bottleneck.decode_tokens(tokens, **kwargs)

        return self.decode(latents, **kwargs)
        
    
    def preprocess_audio_for_encoder(self, audio, in_sr):
        '''
        Preprocess single audio tensor (Channels x Length) to be compatible with the encoder.
        If the model is mono, stereo audio will be converted to mono.
        Audio will be silence-padded to be a multiple of the model's downsampling ratio.
        Audio will be resampled to the model's sample rate. 
        The output will have batch size 1 and be shape (1 x Channels x Length)
        '''
        return self.preprocess_audio_list_for_encoder([audio], [in_sr])

    def preprocess_audio_list_for_encoder(self, audio_list, in_sr_list):
        '''
        Preprocess a [list] of audio (Channels x Length) into a batch tensor to be compatable with the encoder. 
        The audio in that list can be of different lengths and channels. 
        in_sr can be an integer or list. If it's an integer it will be assumed it is the input sample_rate for every audio.
        All audio will be resampled to the model's sample rate. 
        Audio will be silence-padded to the longest length, and further padded to be a multiple of the model's downsampling ratio. 
        If the model is mono, all audio will be converted to mono. 
        The output will be a tensor of shape (Batch x Channels x Length)
        '''
        batch_size = len(audio_list)
        if isinstance(in_sr_list, int):
            in_sr_list = [in_sr_list]*batch_size
        assert len(in_sr_list) == batch_size, "list of sample rates must be the same length of audio_list"
        new_audio = []
        max_length = 0
        # resample & find the max length
        for i in range(batch_size):
            audio = audio_list[i]
            in_sr = in_sr_list[i]
            if len(audio.shape) == 3 and audio.shape[0] == 1:
                # batchsize 1 was given by accident. Just squeeze it.
                audio = audio.squeeze(0)
            elif len(audio.shape) == 1:
                # Mono signal, channel dimension is missing, unsqueeze it in
                audio = audio.unsqueeze(0)
            assert len(audio.shape)==2, "Audio should be shape (Channels x Length) with no batch dimension" 
            # Resample audio
            if in_sr != self.sample_rate:
                resample_tf = T.Resample(in_sr, self.sample_rate).to(audio.device)
                audio = resample_tf(audio)
            new_audio.append(audio)
            if audio.shape[-1] > max_length:
                max_length = audio.shape[-1]
        # Pad every audio to the same length, multiple of model's downsampling ratio
        padded_audio_length = max_length + (self.min_length - (max_length % self.min_length)) % self.min_length
        for i in range(batch_size):
            # Pad it & if necessary, mixdown/duplicate stereo/mono channels to support model
            new_audio[i] = prepare_audio(new_audio[i], in_sr=in_sr, target_sr=in_sr, target_length=padded_audio_length, 
                target_channels=self.in_channels, device=new_audio[i].device).squeeze(0)
        # convert to tensor 
        return torch.stack(new_audio) 

    def encode_audio(self, audio, chunked=False, overlap=32, chunk_size=128, **kwargs):
        '''
        Encode audios into latents. Audios should already be preprocesed by preprocess_audio_for_encoder.
        If chunked is True, split the audio into chunks of a given maximum size chunk_size, with given overlap.
        Overlap and chunk_size params are both measured in number of latents (not audio samples) 
        # and therefore you likely could use the same values with decode_audio. 
        A overlap of zero will cause discontinuity artefacts. Overlap should be => receptive field size. 
        Every autoencoder will have a different receptive field size, and thus ideal overlap.
        You can determine it empirically by diffing unchunked vs chunked output and looking at maximum diff.
        The final chunk may have a longer overlap in order to keep chunk_size consistent for all chunks.
        Smaller chunk_size uses less memory, but more compute.
        The chunk_size vs memory tradeoff isn't linear, and possibly depends on the GPU and CUDA version
        For example, on a A6000 chunk_size 128 is overall faster than 256 and 512 even though it has more chunks
        '''
        if not chunked:
            # default behavior. Encode the entire audio in parallel
            return self.encode(audio, **kwargs)
        else:
            raise NotImplementedError(f'Chunk not support yet')
    
    def decode_audio(self, latents, stds=None, chunked=False, overlap=32, chunk_size=128, **kwargs):
        '''
        Decode latents to audio. 
        If chunked is True, split the latents into chunks of a given maximum size chunk_size, with given overlap, both of which are measured in number of latents. 
        A overlap of zero will cause discontinuity artefacts. Overlap should be => receptive field size. 
        Every autoencoder will have a different receptive field size, and thus ideal overlap.
        You can determine it empirically by diffing unchunked vs chunked audio and looking at maximum diff.
        The final chunk may have a longer overlap in order to keep chunk_size consistent for all chunks.
        Smaller chunk_size uses less memory, but more compute.
        The chunk_size vs memory tradeoff isn't linear, and possibly depends on the GPU and CUDA version
        For example, on a A6000 chunk_size 128 is overall faster than 256 and 512 even though it has more chunks
        '''
        if not chunked:
            # default behavior. Decode the entire latent in parallel
            return self.decode(latents, stds, **kwargs)
        else:
            raise NotImplementedError(f'Chunk not support yet')


        
# AE factories

def create_encoder_from_config(encoder_config: Dict[str, Any]):
    encoder_type = encoder_config.get("type", None)
    assert encoder_type is not None, "Encoder type must be specified"

    if encoder_type == "oobleck":
        encoder = OobleckEncoder(
            **encoder_config["config"]
        )
    else:
        raise ValueError(f"Unknown encoder type {encoder_type}")
    
    requires_grad = encoder_config.get("requires_grad", True)
    if not requires_grad:
        for param in encoder.parameters():
            param.requires_grad = False

    return encoder

def create_decoder_from_config(decoder_config: Dict[str, Any]):
    decoder_type = decoder_config.get("type", None)
    assert decoder_type is not None, "Decoder type must be specified"

    if decoder_type == "oobleck":
        decoder = OobleckDecoder(
            **decoder_config["config"]
        )
    else:
        raise ValueError(f"Unknown decoder type {decoder_type}")
    
    requires_grad = decoder_config.get("requires_grad", True)
    if not requires_grad:
        for param in decoder.parameters():
            param.requires_grad = False

    return decoder

def create_autoencoder_from_config(config: Dict[str, Any]):
    
    ae_config = config["model"]

    encoder = create_encoder_from_config(ae_config["encoder"])
    decoder = create_decoder_from_config(ae_config["decoder"])

    bottleneck = ae_config.get("bottleneck", None)

    latent_dim = ae_config.get("latent_dim", None)
    assert latent_dim is not None, "latent_dim must be specified in model config"
    downsampling_ratio = ae_config.get("downsampling_ratio", None)
    assert downsampling_ratio is not None, "downsampling_ratio must be specified in model config"
    io_channels = ae_config.get("io_channels", None)
    assert io_channels is not None, "io_channels must be specified in model config"
    sample_rate = config.get("sample_rate", None)
    assert sample_rate is not None, "sample_rate must be specified in model config"

    in_channels = ae_config.get("in_channels", None)
    out_channels = ae_config.get("out_channels", None)

    pretransform = ae_config.get("pretransform", None)

    if pretransform is not None:
        pretransform = create_pretransform_from_config(pretransform, sample_rate)

    if bottleneck is not None:
        bottleneck = create_bottleneck_from_config(bottleneck)

    soft_clip = ae_config["decoder"].get("soft_clip", False)

    return AudioAutoencoder(
        encoder,
        decoder,
        io_channels=io_channels,
        latent_dim=latent_dim,
        downsampling_ratio=downsampling_ratio,
        sample_rate=sample_rate,
        bottleneck=bottleneck,
        pretransform=pretransform,
        in_channels=in_channels,
        out_channels=out_channels,
        soft_clip=soft_clip
    )
