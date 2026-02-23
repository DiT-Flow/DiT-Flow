import yaml
import random
import argparse
import os
import torch
import librosa
from tqdm import tqdm
from diffusers import DDIMScheduler
from model.solospeech.conditioners import SoloSpeech_TSR
from utils import save_audio
import shutil
from vae_modules.autoencoder_wrapper import Autoencoder


parser = argparse.ArgumentParser()
parser.add_argument('--device', type=str, default='cuda')
parser.add_argument('--save-path', type=str, default='./672-2830-4446_2_rec.wav')
parser.add_argument('--input_audio', type=str, default='/export/fs05/tcao7/maskGIT/test/source1_reverberent/672-2830-4446_2.wav')
# pre-trained model path
parser.add_argument('--autoencoder-path', type=str, default='/export/fs05/hwang258/SoloSpeech/pretrained_models/vae-200k.ckpt')
parser.add_argument('--eta', type=int, default=0)

parser.add_argument("--num_infer_steps", type=int, default=200)
# model configs
parser.add_argument('--vae-config', type=str, default='/export/fs05/hwang258/SoloSpeech/pretrained_models/config.json')
parser.add_argument('--tsr-config', type=str, default='./config/SoloSpeech-tse-base.yaml')
parser.add_argument('--tsr-ckpt', type=str, default='/export/fs05/hwang258/enhance/SoloSpeech/solospeech/base_solospeech_tse_ckpt/94.pt')
parser.add_argument('--sample-rate', type=int, default=16000)
parser.add_argument('--debug', type=bool, default=False)
# log and random seed
parser.add_argument('--random-seed', type=int, default=2025)
args = parser.parse_args()

with open(args.tsr_config, 'r') as fp:
    args.tsr_config = yaml.safe_load(fp)

args.v_prediction = args.tsr_config["ddim"]["v_prediction"]

@torch.no_grad()
def sample_diffusion(tsr_model, autoencoder, std, scheduler, device,
                     mixture=None, lengths=None, 
                     ddim_steps=50, eta=0, seed=2023
                     ):

    generator = torch.Generator(device=device).manual_seed(seed)
    scheduler.set_timesteps(ddim_steps)
    tsr_pred = torch.randn(mixture.shape, generator=generator, device=device)
    
    for t in scheduler.timesteps:
        tsr_pred = scheduler.scale_model_input(tsr_pred, t)
        model_output, _ = tsr_model(
            x=tsr_pred, 
            timesteps=t, 
            mixture=mixture, 
            x_len=lengths, 
            )
        tsr_pred = scheduler.step(model_output=model_output, timestep=t, sample=tsr_pred,
                                eta=eta, generator=generator).prev_sample

    tsr_pred = autoencoder(embedding=tsr_pred.transpose(2,1), std=std).squeeze(1)

    return tsr_pred



if __name__ == '__main__':

    # os.makedirs(args.output_dir, exist_ok=True)
    autoencoder = Autoencoder(args.autoencoder_path, args.vae_config, 'stft_vae', quantization_first=True)
    autoencoder.eval()
    autoencoder.to(args.device)

    tsr_model = SoloSpeech_TSR(
        args.tsr_config['diffwrap']['UDiT']
    ).to(args.device)
    tsr_model.load_state_dict(torch.load(args.tsr_ckpt)['model'])
    tsr_model.eval()

    total = sum([param.nelement() for param in tsr_model.parameters()])
    print("TSR Number of parameter: %.2fM" % (total / 1e6))
    
    noise_scheduler = DDIMScheduler(**args.tsr_config["ddim"]['diffusers'])
    
    # these steps reset dtype of noise_scheduler params
    latents = torch.randn((1, 128, 128),
                          device=args.device)
    noise = torch.randn(latents.shape).to(latents.device)
    timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps,
                              (noise.shape[0],),
                              device=latents.device).long()
    _ = noise_scheduler.add_noise(latents, noise, timesteps)
    
    
    mixture, _ = librosa.load(args.input_audio, sr=16000)
    with torch.no_grad():
        mixture_input = torch.tensor(mixture).unsqueeze(0).to(args.device)
        mixture_wav = mixture_input
        mixture_input, std = autoencoder(audio=mixture_input.unsqueeze(1))
        lengths = torch.LongTensor([mixture_input.shape[-1]]).to(args.device)
                
        tsr_pred = sample_diffusion(tsr_model, autoencoder, std, noise_scheduler, args.device, mixture_input.transpose(2,1), lengths, ddim_steps=args.num_infer_steps, eta=args.eta, seed=args.random_seed)
        save_audio(args.save_path, 16000, tsr_pred)
