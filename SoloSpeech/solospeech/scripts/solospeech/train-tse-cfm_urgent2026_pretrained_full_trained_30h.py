import yaml
import random
import argparse
import os
import librosa
import soundfile as sf
from tqdm import tqdm
import time

import torch
import torchaudio
from torch.utils.data import DataLoader, ConcatDataset

from accelerate import Accelerator
from diffusers import DDIMScheduler

from model.solospeech.conditioners import SoloSpeech_TSR
from inference import eval_ddim
from dataset import TSEDataset2
from vae_modules.autoencoder_wrapper import Autoencoder
from diffusers import FlowMatchEulerDiscreteScheduler
import pandas as pd
import csv


parser = argparse.ArgumentParser()

# data loading settings
parser.add_argument('--train-clean', type=str, default='/export/fs05/tcao7/urgent2026/simulation_train_resample/clean/')
# parser.add_argument('--train-clean', type=str, default='/export/fs05/tcao7/Storm/wsj0_enh_chime_wv1only/audio/tr/clean/')
parser.add_argument('--train-reverb', type=str, default='/export/fs05/tcao7/urgent2026/simulation_train_resample/noisy/')
# parser.add_argument('--train-reverb', type=str, default='/export/fs05/tcao7/Storm/wsj0_enh_chime_wv1only/audio/tr/noisy/')

parser.add_argument('--sample-rate', type=int, default=16000)
parser.add_argument('--vae-rate', type=int, default=50)
parser.add_argument('--debug', type=bool, default=False)
parser.add_argument('--min-length', type=float, default=3.0)
parser.add_argument("--num-infer-steps", type=int, default=50)
# training settings
parser.add_argument("--amp", type=str, default='fp16')
parser.add_argument('--epochs', type=int, default=400)
parser.add_argument('--batch-size', type=int, default=16)
parser.add_argument('--num-workers', type=int, default=16)
parser.add_argument('--num-threads', type=int, default=1)
parser.add_argument('--save-every', type=int, default=1)
parser.add_argument("--adam-epsilon", type=float, default=1e-08)

# model configs
parser.add_argument('--diffusion-config', type=str, default='./config/base_solospeech_tse_cfm_urgent2026_pretrained_full_trained_30h.yaml')
parser.add_argument('--autoencoder-path', type=str, default='./pretrained_models/audio-vae.pt')
parser.add_argument('--resume-from', type=str, default="/export/fs05/tcao7/enhance/SoloSpeech/solospeech/base_solospeech_tse_cfm_urgent2026_baseline_full_ckpt/pretrained.pt", help='Path to checkpoint to resume training')

# optimization
parser.add_argument('--learning-rate', type=float, default=1e-4)
parser.add_argument('--beta1', type=float, default=0.9)
parser.add_argument('--beta2', type=float, default=0.999)
parser.add_argument('--weight-decay', type=float, default=1e-4)

# log and random seed
parser.add_argument('--random-seed', type=int, default=2025)
parser.add_argument('--log-step', type=int, default=50)
parser.add_argument('--log-dir', type=str, default='logs/')
parser.add_argument('--save-dir', type=str, default='ckpt/')


args = parser.parse_args()

with open(args.diffusion_config, 'r') as fp:
    args.diff_config = yaml.safe_load(fp)


args.v_prediction = args.diff_config["ddim"]["v_prediction"]
# args.log_dir = args.log_dir.replace('log', args.diff_config["system"] + '_log')
# args.save_dir = args.save_dir.replace('ckpt', args.diff_config["system"] + '_ckpt')
args.log_dir = args.log_dir.replace('log', "30hour/" + args.diff_config["system"] + '_log')
args.save_dir = args.save_dir.replace('ckpt', "30hour/" + args.diff_config["system"] + '_ckpt')

if os.path.exists(args.log_dir + '/audio/gt') is False:
    os.makedirs(args.log_dir + '/audio/gt', exist_ok=True)

if os.path.exists(args.save_dir) is False:
    os.makedirs(args.save_dir, exist_ok=True)

    
def masked_mse_loss(predictions, targets, mask=None):
    """
    Computes the masked mean squared error (MSE) loss for tensors of shape (batch_size, sequence_length, feature_size).
    
    Args:
        predictions (torch.Tensor): The model's predictions of shape (batch_size, sequence_length, feature_size).
        targets (torch.Tensor): The ground truth values of the same shape as predictions.
        mask (torch.Tensor): A boolean mask of shape (batch_size, sequence_length) indicating which sequences to include.
    
    Returns:
        torch.Tensor: The masked MSE loss.
    """

    if mask is not None:
        mask = mask.unsqueeze(-1).long()
        mse = (predictions - targets) ** 2
        masked_mse = mse * mask
        loss = masked_mse.sum() / mask.sum()
    else:
        mse = (predictions - targets) ** 2
        loss = mse.mean()
        
    return loss

def load_candidates(meta_tsv):
    """Load rows where duration<20s and augmentation!='none'."""
    candidates = []
    with open(meta_tsv, newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for r in reader:
            try:
                fs = float(r["fs"])
                length = float(r["length"])
                dur = length / fs if fs > 0 else float("inf")
            except Exception:
                continue

            aug = (r.get("augmentation") or "").strip().lower()
            if dur < 20.0 and aug != "none":
                candidates.append({
                    "id": r.get("id"),
                    "duration": dur,
                    "noisy_path": r.get("noisy_path"),
                    "clean_path": r.get("clean_path"),
                    "augmentation": r.get("augmentation"),
                })
    # ensure uniqueness by id (in case of duplicates)
    uniq = {}
    for c in candidates:
        if c["id"] not in uniq:
            uniq[c["id"]] = c
    return list(uniq.values())


def random_pack(candidates, target_seconds, tol_seconds, seed=2026):
    """
    Randomly shuffle then greedily accumulate until within tolerance or just above target.
    Returns the selected list and the total duration.
    """
    rng = random.Random(seed)
    pool = candidates[:]  # copy
    rng.shuffle(pool)

    selected = []
    total = 0.0
    for c in pool:
        # never add same id twice; pool is unique so this check is just defensive
        if any(c["id"] == s["id"] for s in selected):
            continue
        # add and check
        selected.append(c)
        total += c["duration"]
        if total >= target_seconds - tol_seconds:
            # we're at or near the target; if also not overshooting too much, stop
            if abs(total - target_seconds) <= tol_seconds or total >= target_seconds:
                break

    # If still below target (not enough data), return everything we have.
    return selected, total

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


    # read meta.tsv file and remove files >= 20s
    # meta_path = "/home/tcao7/urgent2026_challenge_track1/meta.tsv"
    # meta_path = "/export/fs05/tcao7/urgent2026/simulation_train/log/meta.tsv"
    # df = pd.read_csv(meta_path, sep="\t", dtype=str)

    # # numeric duration
    # df["fs"] = pd.to_numeric(df["fs"], errors="coerce")
    # df["length"] = pd.to_numeric(df["length"], errors="coerce")
    # df["duration_s"] = df["length"] / df["fs"]

    # # filters
    # mask = (
    #     (df["duration_s"] < 20.0) &
    #     (df["augmentation"].fillna("").str.strip().str.lower() == "none")
    # )

    # # choose which paths you want
    # noisy_paths = df.loc[mask, "noisy_path"].dropna().tolist()
    # noisy_paths = sorted([item.replace("simulation_train", "simulation_train_resample") for item in noisy_paths])
    # clean_paths = df.loc[mask, "clean_path"].dropna().tolist()
    # clean_paths = sorted([item.replace("simulation_train", "simulation_train_resample") for item in clean_paths])
    
    META_TSV = "/export/fs05/tcao7/urgent2026/simulation_train/log/meta.tsv"
    TARGET_HOURS = 30.0             # select 12-hour data to finetune
    TOLERANCE_SECONDS = 300         # how close to target to aim for (±5 min)
    SEED = 2026                     # change for a different random selection

    target_seconds = TARGET_HOURS * 3600.0
    candidates = load_candidates(META_TSV)

    if not candidates:
        raise SystemExit("No candidates found with duration<20s and augmentation!='none'.")

    selected, total = random_pack(candidates, target_seconds, TOLERANCE_SECONDS, SEED)

    noisy_paths = sorted([s["noisy_path"] for s in selected if s["noisy_path"]])
    noisy_paths = [item.replace("simulation_train", "simulation_train_resample") for item in noisy_paths]
    clean_paths = sorted([s["clean_path"] for s in selected if s["clean_path"]])
    clean_paths = [item.replace("simulation_train", "simulation_train_resample") for item in clean_paths]

    print(f"kept {len(noisy_paths)} noisy files and {len(clean_paths)} clean files")




    
    train_set = TSEDataset2(
        reverb_dir=noisy_paths, 
        clean_dir=clean_paths,
        debug=args.debug,
    )
    train_loader = DataLoader(train_set, num_workers=args.num_workers, batch_size=args.batch_size, shuffle=True, pin_memory=True, collate_fn=train_set.collate)


    # use accelerator for multi-gpu training
    accelerator = Accelerator(mixed_precision=args.amp)
    
    model = SoloSpeech_TSR(
        args.diff_config['diffwrap']['UDiT']
    )

    total = sum([param.nelement() for param in model.parameters()])
    print("Number of parameter: %.2fM" % (total / 1e6))
    

    noise_scheduler = FlowMatchEulerDiscreteScheduler(**args.diff_config["ddim"]['diffusers'])


    optimizer = torch.optim.AdamW(model.parameters(),
                                  lr=args.learning_rate,
                                  betas=(args.beta1, args.beta2),
                                  weight_decay=args.weight_decay,
                                  eps=args.adam_epsilon,
                                  )

    if args.resume_from is not None and os.path.exists(args.resume_from):
        checkpoint = torch.load(args.resume_from, map_location='cpu')
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        global_step = checkpoint["global_step"]
        start_epoch = checkpoint["epoch"] + 1  # Continue from the next epoch
        print(f"Resuming training from checkpoint: {args.resume_from}, starting from epoch {start_epoch}.")
    else:
        global_step = 0
        start_epoch = 0
    
    model.to(accelerator.device)
    model, optimizer, train_loader = accelerator.prepare(model, optimizer, train_loader)

    losses = 0

    for epoch in range(start_epoch, args.epochs):
        model.train()
        for step, batch in enumerate(tqdm(train_loader)):
            # compress by vae
            clean, reverb, lengths = batch['clean_vae'], batch['reverb_vae'], batch['length']
            clean = clean.to(accelerator.device)
            reverb = reverb.to(accelerator.device)
            lengths = lengths.to(accelerator.device)
            
            # adding noise
            noise = torch.randn(clean.shape).to(accelerator.device)
            # timesteps = torch.randint(0, noise_scheduler.num_train_timesteps, (noise.shape[0],),
            #                           device=accelerator.device,).long()
            # noisy_target = noise_scheduler.add_noise(clean, noise, timesteps)
            # # v prediction - model output
            # velocity = noise_scheduler.get_velocity(clean, noise, timesteps)
            sigmas = torch.rand((clean.shape[0],), dtype=clean.dtype, device=clean.device)
            timesteps = sigmas * 1000
            while len(sigmas.shape) < clean.ndim:
                sigmas = sigmas.unsqueeze(-1)
            noisy_target = sigmas * noise.clone() + (1.0 - sigmas) * clean.clone()
            # flow matching velocity
            velocity = noise.clone() - clean.clone()
            # inference
            pred, pred_mask = model(x=noisy_target, timesteps=timesteps, mixture=reverb, x_len=lengths)
            # backward
            if args.v_prediction:
                loss = masked_mse_loss(pred, velocity, pred_mask)
            else:
                loss = masked_mse_loss(pred, noise, pred_mask)

            is_nan = torch.isnan(loss).item()
            if not is_nan: #skip nan loss
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
            else:
                torch.cuda.empty_cache()
                n = open(args.log_dir + 'log.txt', mode='a')
                n.write(time.asctime(time.localtime(time.time())))
                n.write('\n')
                n.write('Epoch: [{}][{}]    Batch: [{}][{}]  Nan  Loss\n'.format(
                    epoch + 1, args.epochs, step+1, len(train_loader)))
                n.close()

        if accelerator.is_main_process:
            if (epoch + 1) % args.save_every == 0:
                accelerator.wait_for_everyone()
                unwrapped_model = accelerator.unwrap_model(model)
                accelerator.save({
                    "model": unwrapped_model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "epoch": epoch,
                    "global_step": global_step,
                }, args.save_dir+str(epoch)+'.pt')
        accelerator.wait_for_everyone()
