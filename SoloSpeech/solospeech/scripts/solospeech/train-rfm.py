import yaml
import random
import argparse
import os
import librosa
import soundfile as sf
from tqdm import tqdm
import time
import copy

import torch
import torchaudio
from torch.utils.data import DataLoader, ConcatDataset

from accelerate import Accelerator
# from diffusers import DDIMScheduler
from diffusers import FlowMatchEulerDiscreteScheduler
from diffusers.training_utils import (
    compute_density_for_timestep_sampling,
    compute_loss_weighting_for_sd3,
)

from model.udit import UDiT
from utils import save_plot, save_audio, minmax_norm_diff, reverse_minmax_norm_diff
from inference_rfm import eval_ddim
from dataset import TSEDataset
from vae_modules.autoencoder_wrapper import Autoencoder

parser = argparse.ArgumentParser()

# data loading settings
parser.add_argument('--train-csv-dir', type=str, default='/export/corpora7/HW/speakerbeam-main/egs/libri2mix/data/wav16k/min/train-360')
parser.add_argument('--val-csv-dir', type=str, default='/export/corpora7/HW/speakerbeam-main/egs/libri2mix/data/wav16k/min/dev')
parser.add_argument('--base-dir', type=str, default='/export/corpora7/HW/LibriMix')
parser.add_argument('--vae-dir', type=str, default='/export/corpora7/HW/audio-vae-16k/stable_vae_emb')
parser.add_argument('--wavlmsv-dir', type=str, default='/export/corpora7/HW/audio-vae/wavlmplussv_emb')
parser.add_argument('--wavlmplusall-dir', type=str, default='/export/corpora7/HW/audio-vae-16k/wavlmplusall_emb')
parser.add_argument('--ecapatdnn-dir', type=str, default='/export/corpora7/HW/audio-vae/ecapa_tdnn_emb')
parser.add_argument('--wavlm-dir', type=str, default='/export/corpora7/HW/audio-vae-16k/wavlm_emb')
parser.add_argument('--uncond-wavlmsv', type=str, default='/export/corpora7/HW/SoloSpeech-main/pretrained_models/uncond_wavlmsv.pt')
parser.add_argument('--uncond-ecapatdnn', type=str, default='/export/corpora7/HW/SoloSpeech-main/pretrained_models/uncond_ecapatdnn.pt')
parser.add_argument('--uncond-wavlmplusall', type=str, default='/export/corpora7/HW/SoloSpeech-main/pretrained_models/uncond_wavlmplusall.pt')

parser.add_argument('--sample-rate', type=int, default=16000)
parser.add_argument('--vae-rate', type=int, default=50)
parser.add_argument('--debug', type=bool, default=False)
parser.add_argument('--use-wavlmplusall', type=bool, default=False)
parser.add_argument('--timbre-prompt-length', type=int, default=3)
parser.add_argument('--min-length', type=float, default=3.0)
parser.add_argument("--cfg-ratio", type=float, default=0.1)
parser.add_argument("--num-infer-steps", type=int, default=50)
parser.add_argument('--use-wavlm', type=bool, default=True)
parser.add_argument('--use-target', type=bool, default=True)
parser.add_argument("--align-ratio", type=float, default=1.0)
# training settings
parser.add_argument("--amp", type=str, default='fp16')
parser.add_argument('--epochs', type=int, default=300)
parser.add_argument('--batch-size', type=int, default=128)
parser.add_argument('--num-workers', type=int, default=16)
parser.add_argument('--num-threads', type=int, default=1)
parser.add_argument('--save-every', type=int, default=5)
parser.add_argument("--adam-epsilon", type=float, default=1e-08)

# model configs
parser.add_argument('--diffusion-config', type=str, default='./config/SoloSpeech_rfm.yaml')
parser.add_argument('--autoencoder-path', type=str, default='./pretrained_models/audio-vae.pt')

# optimization
parser.add_argument('--learning-rate', type=float, default=1e-4)
parser.add_argument('--beta1', type=float, default=0.9)
parser.add_argument('--beta2', type=float, default=0.999)
parser.add_argument('--weight-decay', type=float, default=1e-4)

# log and random seed
parser.add_argument('--random-seed', type=int, default=2024)
parser.add_argument('--log-step', type=int, default=50)
parser.add_argument('--log-dir', type=str, default='logs/')
parser.add_argument('--save-dir', type=str, default='ckpt/')


args = parser.parse_args()

with open(args.diffusion_config, 'r') as fp:
    args.diff_config = yaml.safe_load(fp)


args.v_prediction = args.diff_config["ddim"]["v_prediction"]
args.log_dir = args.log_dir.replace('log', args.diff_config["system"] + '_log')
args.save_dir = args.save_dir.replace('ckpt', args.diff_config["system"] + '_ckpt')

if os.path.exists(args.log_dir + '/audio/gt') is False:
    os.makedirs(args.log_dir + '/audio/gt', exist_ok=True)

if os.path.exists(args.save_dir) is False:
    os.makedirs(args.save_dir, exist_ok=True)

    
def masked_mse_loss(predictions, targets, lengths=None):
    """
    Computes the masked mean squared error (MSE) loss for tensors of shape (batch_size, sequence_length, feature_size).
    
    Args:
        predictions (torch.Tensor): The model's predictions of shape (batch_size, sequence_length, feature_size).
        targets (torch.Tensor): The ground truth values of the same shape as predictions.
        mask (torch.Tensor): A boolean mask of shape (batch_size, sequence_length) indicating which sequences to include.
    
    Returns:
        torch.Tensor: The masked MSE loss.
    """

    if lengths is not None:
        max_len = lengths.max()  # Maximum length to pad to
        range_tensor = torch.arange(max_len).expand(len(lengths), max_len).to(max_len.device)
        mask = range_tensor < lengths.unsqueeze(1)
        mask = mask.unsqueeze(-1).long()
        mse = (predictions - targets) ** 2
        masked_mse = mse * mask
        loss = masked_mse.sum() / mask.sum()
    else:
        mse = (predictions - targets) ** 2
        loss = mse.mean()
        
    return loss

if __name__ == '__main__':
    # Fix the random seed
    random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)

    # Set device
    torch.set_num_threads(args.num_threads)
    if torch.cuda.is_available():
        args.device = 'cuda'
        torch.cuda.manual_seed(args.random_seed)
        torch.cuda.manual_seed_all(args.random_seed)
        torch.backends.cuda.matmul.allow_tf32 = True
        if torch.backends.cudnn.is_available():
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    else:
        args.device = 'cpu'
        
    args.use_wavlmplusall = args.diff_config['diffwrap']['UDiT']['use_wavlmplusall']
    args.use_wavlm = args.diff_config['diffwrap']['UDiT']['use_wavlm']
    args.use_wavlmsv = args.diff_config['diffwrap']['UDiT']['use_wavlmsv']
    args.use_ecapatdnn = args.diff_config['diffwrap']['UDiT']['use_ecapatdnn']
    args.use_target = args.diff_config['diffwrap']['UDiT']['use_target']
    
    train_set = TSEDataset(
        csv_dir=args.train_csv_dir, 
        base_dir=args.base_dir, 
        vae_dir=args.vae_dir, 
        wavlmsv_dir=args.wavlmsv_dir,
        wavlmplusall_dir=args.wavlmplusall_dir,
        ecapatdnn_dir=args.ecapatdnn_dir,
        wavlm_dir=args.wavlm_dir,
        use_wavlmplusall=args.use_wavlmplusall,
        use_wavlm=args.use_wavlm,
        use_wavlmsv=args.use_wavlmsv,
        use_ecapatdnn=args.use_ecapatdnn,
        use_target=args.use_target,
        task="sep_noisy", 
        sample_rate=args.sample_rate, 
        vae_rate=args.vae_rate,
        n_src=2, 
        min_length=args.min_length,
        debug=args.debug,
        timbre_prompt_length=args.timbre_prompt_length,
        uncond_wavlmsv=args.uncond_wavlmsv,
        uncond_ecapatdnn=args.uncond_ecapatdnn,
        uncond_wavlmplusall=args.uncond_wavlmplusall,
        training=True, 
        cfg_ratio=args.cfg_ratio,
    )
    train_loader = DataLoader(train_set, num_workers=args.num_workers, batch_size=args.batch_size, shuffle=True, pin_memory=True, persistent_workers=True, collate_fn=train_set.collate)

    # use this load for check generated audio samples
    eval_set = TSEDataset(
        csv_dir=args.val_csv_dir, 
        base_dir=args.base_dir, 
        vae_dir=args.vae_dir, 
        wavlmsv_dir=args.wavlmsv_dir,
        wavlmplusall_dir=args.wavlmplusall_dir,
        ecapatdnn_dir=args.ecapatdnn_dir,
        wavlm_dir=args.wavlm_dir,
        use_wavlmplusall=args.use_wavlmplusall,
        use_wavlm=args.use_wavlm,
        use_wavlmsv=args.use_wavlmsv,
        use_ecapatdnn=args.use_ecapatdnn,
        use_target=args.use_target,
        task="sep_noisy", 
        sample_rate=args.sample_rate, 
        vae_rate=args.vae_rate,
        n_src=2, 
        min_length=args.min_length,
        debug=True,
        timbre_prompt_length=args.timbre_prompt_length,
        uncond_wavlmsv=args.uncond_wavlmsv,
        uncond_ecapatdnn=args.uncond_ecapatdnn,
        uncond_wavlmplusall=args.uncond_wavlmplusall,
        training=False, 
        cfg_ratio=0.0,
    )
    eval_loader = DataLoader(eval_set, num_workers=args.num_workers, batch_size=1, shuffle=False, pin_memory=True, persistent_workers=True, collate_fn=eval_set.collate)
    # use these two loaders for benchmarks

    # use accelerator for multi-gpu training
    accelerator = Accelerator(mixed_precision=args.amp)
    
    unet = UDiT(
        **args.diff_config['diffwrap']['UDiT']
    ).to(accelerator.device)

    total = sum([param.nelement() for param in unet.parameters()])
    print("Number of parameter: %.2fM" % (total / 1e6))
    
    autoencoder = Autoencoder(args.autoencoder_path, 'stable_vae', quantization_first=True)
    autoencoder.eval()
    autoencoder.to(accelerator.device)


    noise_scheduler = FlowMatchEulerDiscreteScheduler(**args.diff_config["ddim"]['diffusers'])
    noise_scheduler_eval = copy.deepcopy(noise_scheduler)

    optimizer = torch.optim.AdamW(unet.parameters(),
                                  lr=args.learning_rate,
                                  betas=(args.beta1, args.beta2),
                                  weight_decay=args.weight_decay,
                                  eps=args.adam_epsilon,
                                  )

    unet, autoencoder, optimizer, train_loader = accelerator.prepare(unet, autoencoder, optimizer, train_loader)

    global_step = 0
    losses = 0
    
    if accelerator.is_main_process:
        unet_module = unet.module if hasattr(unet, 'module') else unet
        eval_ddim(unet_module, autoencoder, noise_scheduler_eval, eval_loader, args, accelerator.device, epoch='test', ddim_steps=args.num_infer_steps, eta=0, guidance_scale=2.5)
    accelerator.wait_for_everyone()

    for epoch in range(args.epochs):
        unet.train()
        for step, batch in enumerate(tqdm(train_loader)):
            # compress by vae
            mixture, target, wavlmsv, ecapatdnn, lengths, wavlmplusall, mixture_wavlm = batch['mixture_vae'], batch['source_vae'], batch['wavlmsv'], batch['ecapatdnn'], batch['length'], batch['wavlmplusall'], batch['mixture_wavlm']
            mixture = mixture.to(accelerator.device)
            target = target.to(accelerator.device)
            lengths = lengths.to(accelerator.device)
            if args.use_wavlmsv:
                wavlmsv = wavlmsv.to(accelerator.device)
            if args.use_ecapatdnn:
                ecapatdnn = ecapatdnn.to(accelerator.device)
            if args.use_wavlm:
                mixture_wavlm = mixture_wavlm.to(accelerator.device)
            if args.use_wavlmplusall:
                wavlmplusall = wavlmplusall.to(accelerator.device)

            # adding noise
            noise = torch.randn(target.shape).to(accelerator.device)
            
            # timesteps = torch.randint(0, noise_scheduler.num_train_timesteps, (noise.shape[0],),
            #                           device=accelerator.device,).long()
            # noisy_target = noise_scheduler.add_noise(target, noise, timesteps)
            # # v prediction - model output
            # velocity = noise_scheduler.get_velocity(target, noise, timesteps)

            sigmas = torch.rand((audio_clip.shape[0],), dtype=audio_clip.dtype, device=audio_clip.device)
            timesteps = sigmas * 1000
            while len(sigmas.shape) < audio_clip.ndim:
                sigmas = sigmas.unsqueeze(-1)
            noisy_target = sigmas * noise.clone() + (1.0 - sigmas) * target.clone()
            # flow matching velocity
            velocity = noise.clone() - target.clone()
            
            # inference
            pred, semantic = unet(x=noisy_target, timesteps=timesteps, mixture=mixture, wavlmplusall_timbre=wavlmplusall, wavlmsv_timbre=wavlmsv, ecapatdnn_timbre=ecapatdnn, lengths=lengths, mixture_wavlm=mixture_wavlm)
            # backward
            if args.v_prediction:
                if args.use_target:
                    loss = masked_mse_loss(pred, velocity, lengths) + args.align_ratio * masked_mse_loss(semantic, target, lengths)
                else:
                    loss = masked_mse_loss(pred, velocity, lengths)
            else:
                if args.use_target:
                    loss = masked_mse_loss(pred, noise, lengths) + args.align_ratio * masked_mse_loss(semantic, target, lengths)
                else:
                    loss = masked_mse_loss(pred, noise, lengths)
                
            accelerator.backward(loss)
            optimizer.step()
            optimizer.zero_grad()

            global_step += 1
            losses += loss.item()

            if accelerator.is_main_process:
                if global_step % args.log_step == 0:
                    n = open(args.log_dir + 'log.txt', mode='a')
                    n.write(time.asctime(time.localtime(time.time())))
                    n.write('\n')
                    n.write('Epoch: [{}][{}]    Batch: [{}][{}]    Loss: {:.6f}\n'.format(
                        epoch + 1, args.epochs, step+1, len(train_loader), losses / args.log_step))
                    n.close()
                    losses = 0.0

        if accelerator.is_main_process:
            unet_module = unet.module if hasattr(unet, 'module') else unet
            eval_ddim(unet_module, autoencoder, noise_scheduler_eval, eval_loader, args, accelerator.device, epoch=epoch+1, ddim_steps=args.num_infer_steps, eta=0, guidance_scale=2.5)

            if (epoch + 1) % args.save_every == 0:
                accelerator.wait_for_everyone()
                unwrapped_unet = accelerator.unwrap_model(unet)
                accelerator.save({
                    "model": unwrapped_unet.state_dict(),
                }, args.save_dir+str(epoch)+'.pt')
        accelerator.wait_for_everyone()