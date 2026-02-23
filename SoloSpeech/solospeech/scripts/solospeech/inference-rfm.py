import torch
import os
from utils import save_audio, get_loss
from tqdm import tqdm
import shutil
import numpy as np

def rescale_noise_cfg(noise_cfg, noise_pred_text, guidance_rescale=0.0):
    """
    Rescale `noise_cfg` according to `guidance_rescale`. Based on findings of [Common Diffusion Noise Schedules and
    Sample Steps are Flawed](https://arxiv.org/pdf/2305.08891.pdf). See Section 3.4
    """
    std_text = noise_pred_text.std(dim=list(range(1, noise_pred_text.ndim)), keepdim=True)
    std_cfg = noise_cfg.std(dim=list(range(1, noise_cfg.ndim)), keepdim=True)
    # rescale the results from guidance (fixes overexposure)
    noise_pred_rescaled = noise_cfg * (std_text / std_cfg)
    # mix with the original results from guidance by factor guidance_rescale to avoid "plain looking" images
    noise_cfg = guidance_rescale * noise_pred_rescaled + (1 - guidance_rescale) * noise_cfg
    return noise_cfg


@torch.no_grad()
def eval_ddim(unet, autoencoder, scheduler, eval_loader, args, device, epoch=0, 
              guidance_scale=False, guidance_rescale=0.0,
              ddim_steps=50, eta=0, 
              random_seed=2024,):
    
    if random_seed is not None:
        generator = torch.Generator(device=device).manual_seed(random_seed)
    else:
        generator = torch.Generator(device=device)
        generator.seed()
    scheduler.set_timesteps(ddim_steps)
    unet.eval()

    for step, batch in enumerate(tqdm(eval_loader)):
        mixture, target, wavlmsv, ecapatdnn, lengths, wavlmplusall, mixture_wavlm = batch['mixture_vae'], batch['source_vae'], batch['wavlmsv'], batch['ecapatdnn'], batch['length'], batch['wavlmplusall'], batch['mixture_wavlm']
        mixture = mixture.to(device)
        target = target.to(device)
        lengths = lengths.to(device)
        mixture_path, source_path, enroll_path, mix_id = batch['mixture_path'], batch['source_path'], batch['enroll_path'], batch['id']
        if args.use_wavlmsv:
            wavlmsv = wavlmsv.to(device)
        if args.use_wavlmplusall:
            wavlmplusall = wavlmplusall.to(device)
        if args.use_wavlm:
            mixture_wavlm = mixture_wavlm.to(device)
        if args.use_ecapatdnn:
            ecapatdnn = ecapatdnn.to(device)
        # init noise
        noise = torch.randn(mixture.shape, generator=generator, device=device)
        pred = noise

        for t in scheduler.timesteps:
            # pred = scheduler.scale_model_input(pred, t)
            if guidance_scale:
                pred_combined = torch.cat([pred, pred], dim=0)
                mixture_combined = torch.cat([mixture, mixture], dim=0)
                lengths_combined = torch.cat([lengths, lengths], dim=0)
                
                wavlmplusall_combined = None
                if args.use_wavlmplusall:
                    uncond_wavlmplusall = torch.load(args.uncond_wavlmplusall).squeeze()
                    uncond_wavlmplusall = uncond_wavlmplusall.transpose(2, 0)
                    target_length = int(args.timbre_prompt_length * args.vae_rate)
                    uncond_wavlmplusall = uncond_wavlmplusall[:, :target_length]
                    uncond_wavlmplusall = uncond_wavlmplusall.unsqueeze(0).to(device)
                    wavlmplusall_combined = torch.cat([wavlmplusall, uncond_wavlmplusall], dim=0)
                mixture_wavlm_combined = None
                if args.use_wavlm:
                    mixture_wavlm_combined = torch.cat([mixture_wavlm, mixture_wavlm], dim=0)
                wavlmsv_combined = None
                if args.use_wavlmsv:
                    uncond_wavlmsv = torch.load(args.uncond_wavlmsv).squeeze().unsqueeze(0).to(device)
                    wavlmsv_combined = torch.cat([wavlmsv, uncond_wavlmsv], dim=0)
                ecapatdnn_combined = None
                if args.use_ecapatdnn:
                    uncond_ecapatdnn = torch.load(args.uncond_ecapatdnn).squeeze().unsqueeze(0).to(device)
                    ecapatdnn_combined = torch.cat([ecapatdnn, uncond_ecapatdnn], dim=0)
                    
                output_combined, _ = unet(x=pred_combined, timesteps=t, mixture=mixture_combined, wavlmplusall_timbre=wavlmplusall_combined, wavlmsv_timbre=wavlmsv_combined, ecapatdnn_timbre=ecapatdnn_combined, lengths=lengths_combined, mixture_wavlm=mixture_wavlm_combined)
                output_pos, output_neg = torch.chunk(output_combined, 2, dim=0)
    
                model_output = output_neg + guidance_scale * (output_pos - output_neg)
                if guidance_rescale > 0.0:
                    # avoid overexposed
                    model_output = rescale_noise_cfg(model_output, output_pos,
                                                     guidance_rescale=guidance_rescale)
            else:
                model_output, _ = unet(x=pred, timesteps=t, mixture=mixture, wavlmplusall_timbre=wavlmplusall, wavlmsv_timbre=wavlmsv, ecapatdnn_timbre=ecapatdnn, lengths=lengths, mixture_wavlm=mixture_wavlm)
        
            pred = scheduler.step(model_output=model_output, timestep=t, sample=pred,
                                  # eta=eta, generator=generator
                                 ).prev_sample

        pred_wav = autoencoder(embedding=pred.transpose(2, 1))

        os.makedirs(f'{args.log_dir}/audio/{epoch}/', exist_ok=True)

        for j in range(pred_wav.shape[0]):
            length = lengths[j]*(args.sample_rate//args.vae_rate) # 320 upsampling rate
            tmp = pred_wav[j][:, :length].unsqueeze(0)
            shutil.copyfile(mixture_path[j], f'{args.log_dir}/audio/{epoch}/pred_{mix_id[j]}_mixture.wav')
            shutil.copyfile(source_path[j], f'{args.log_dir}/audio/{epoch}/pred_{mix_id[j]}_source.wav')
            shutil.copyfile(enroll_path[j], f'{args.log_dir}/audio/{epoch}/pred_{mix_id[j]}_enroll.wav')
            save_audio(f'{args.log_dir}/audio/{epoch}/pred_{mix_id[j]}.wav', 16000, tmp)