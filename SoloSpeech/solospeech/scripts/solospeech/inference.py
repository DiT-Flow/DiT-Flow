import torch
import os
from utils import save_audio
from tqdm import tqdm
import shutil

@torch.no_grad()
def eval_ddim(model, autoencoder, scheduler, eval_loader, args, device, epoch=0, 
              ddim_steps=50, eta=0, 
              random_seed=2024,):
    
    if random_seed is not None:
        generator = torch.Generator(device=device).manual_seed(random_seed)
    else:
        generator = torch.Generator(device=device)
        generator.seed()
    scheduler.set_timesteps(ddim_steps)
    model.eval()

    for step, batch in enumerate(tqdm(eval_loader)):
        mixture_path, source_path, reference_path, exclude_path, mix_id = batch['mixture_path'], batch['source_path'], batch['reference_path'], batch['exclude_path'], batch['id']
        mixture, target, reference, lengths, reference_lengths = batch['mixture_vae'], batch['source_vae'], batch['reference_vae'], batch['length'], batch['reference_length']
        mixture = mixture.to(device)
        target = target.to(device)
        lengths = lengths.to(device)
        reference = reference.to(device)
        reference_lengths = reference_lengths.to(device)
        # init noise
        noise = torch.randn(mixture.shape, generator=generator, device=device)
        pred = noise

        for t in scheduler.timesteps:
            pred = scheduler.scale_model_input(pred, t)
            model_output, _ = model(
                x=pred, 
                timesteps=t, 
                mixture=mixture, 
                reference=reference, 
                x_len=lengths, 
                ref_len=reference_lengths, 
                )
        
            pred = scheduler.step(model_output=model_output, timestep=t, sample=pred,
                                  eta=eta, generator=generator).prev_sample

        pred_wav = autoencoder(embedding=pred.transpose(2, 1))

        os.makedirs(f'{args.log_dir}/audio/{epoch}/', exist_ok=True)

        for j in range(pred_wav.shape[0]):
            length = lengths[j]*(args.sample_rate//args.vae_rate) # 320 upsampling rate
            tmp = pred_wav[j][:, :length].unsqueeze(0)
            shutil.copyfile(mixture_path[j], f'{args.log_dir}/audio/{epoch}/{mix_id[j]}_mixture.wav')
            shutil.copyfile(source_path[j], f'{args.log_dir}/audio/{epoch}/{mix_id[j]}_source.wav')
            shutil.copyfile(reference_path[j], f'{args.log_dir}/audio/{epoch}/{mix_id[j]}_reference.wav')
            shutil.copyfile(exclude_path[j], f'{args.log_dir}/audio/{epoch}/{mix_id[j]}_exclude.wav')
            save_audio(f'{args.log_dir}/audio/{epoch}/{mix_id[j]}_pred.wav', 16000, tmp)

@torch.no_grad()
def eval_tsr(model, autoencoder, scheduler, eval_loader, args, device, epoch=0, 
              ddim_steps=50, eta=0, 
              random_seed=2024,):
    
    if random_seed is not None:
        generator = torch.Generator(device=device).manual_seed(random_seed)
    else:
        generator = torch.Generator(device=device)
        generator.seed()
    scheduler.set_timesteps(ddim_steps)
    model.eval()

    for step, batch in enumerate(tqdm(eval_loader)):
        mixture_path, source_path, exclude_path, mix_id = batch['mixture_path'], batch['source_path'], batch['exclude_path'], batch['id']
        mixture, target, exclude, lengths = batch['mixture_vae'], batch['source_vae'], batch['exclude_vae'], batch['length']
        mixture = mixture.to(device)
        target = target.to(device)
        exclude = exclude.to(device)
        lengths = lengths.to(device)
        # init noise
        noise = torch.randn(mixture.shape, generator=generator, device=device)
        pred = noise

        for t in scheduler.timesteps:
            pred = scheduler.scale_model_input(pred, t)
            model_output, _ = model(
                x=pred, 
                timesteps=t, 
                mixture=mixture, 
                reference=exclude, 
                x_len=lengths, 
                )
        
            pred = scheduler.step(model_output=model_output, timestep=t, sample=pred,
                                  eta=eta, generator=generator).prev_sample

        pred_wav = autoencoder(embedding=pred.transpose(2, 1))

        os.makedirs(f'{args.log_dir}/audio/{epoch}/', exist_ok=True)

        for j in range(pred_wav.shape[0]):
            length = lengths[j]*(args.sample_rate//args.vae_rate) # 320 upsampling rate
            tmp = pred_wav[j][:, :length].unsqueeze(0)
            shutil.copyfile(mixture_path[j], f'{args.log_dir}/audio/{epoch}/{mix_id[j]}_mixture.wav')
            shutil.copyfile(source_path[j], f'{args.log_dir}/audio/{epoch}/{mix_id[j]}_source.wav')
            shutil.copyfile(exclude_path[j], f'{args.log_dir}/audio/{epoch}/{mix_id[j]}_exclude.wav')
            save_audio(f'{args.log_dir}/audio/{epoch}/{mix_id[j]}_pred.wav', 16000, tmp)


@torch.no_grad()
def eval_disc(model, autoencoder, eval_loader, args, device, epoch=0, 
              random_seed=2024,):
    
    model.eval()

    for step, batch in enumerate(tqdm(eval_loader)):
        mixture_path, source_path, reference_path, exclude_path, mix_id = batch['mixture_path'], batch['source_path'], batch['reference_path'], batch['exclude_path'], batch['id']
        mixture, target, reference, lengths, reference_lengths = batch['mixture_vae'], batch['source_vae'], batch['reference_vae'], batch['length'], batch['reference_length']
        mixture = mixture.to(device)
        target = target.to(device)
        lengths = lengths.to(device)
        reference = reference.to(device)
        reference_lengths = reference_lengths.to(device)
    
        pred, _ = model(
            x=mixture, 
            reference=reference, 
            x_len=lengths, 
            ref_len=reference_lengths, 
            )

        pred_wav = autoencoder(embedding=pred.transpose(2, 1))

        os.makedirs(f'{args.log_dir}/audio/{epoch}/', exist_ok=True)

        for j in range(pred_wav.shape[0]):
            length = lengths[j]*(args.sample_rate//args.vae_rate) # 320 upsampling rate
            tmp = pred_wav[j][:, :length].unsqueeze(0)
            shutil.copyfile(mixture_path[j], f'{args.log_dir}/audio/{epoch}/{mix_id[j]}_mixture.wav')
            shutil.copyfile(source_path[j], f'{args.log_dir}/audio/{epoch}/{mix_id[j]}_source.wav')
            shutil.copyfile(reference_path[j], f'{args.log_dir}/audio/{epoch}/{mix_id[j]}_reference.wav')
            shutil.copyfile(exclude_path[j], f'{args.log_dir}/audio/{epoch}/{mix_id[j]}_exclude.wav')
            save_audio(f'{args.log_dir}/audio/{epoch}/{mix_id[j]}_pred.wav', 16000, tmp)
