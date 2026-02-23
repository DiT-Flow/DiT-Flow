import os
import pandas as pd
from pathlib import Path
from collections import defaultdict
from torch.utils.data import Dataset
import random
import torch
import numpy as np
from glob import glob
import typing

from typing import Optional, Dict, List, Tuple

class TSEDataset2(Dataset):
    def __init__(
            self, 
            reverb_dir, 
            clean_dir, 
            debug=False,
        ):
        self.debug = debug
        # self.clean_files = sorted(glob(os.path.join(clean_dir, '*.npz')))
        # self.reverb_files = [item.replace(clean_dir, reverb_dir) for item in self.clean_files]
        self.clean_files = sorted([item.replace(".flac", ".npz") for item in clean_dir])
        self.reverb_files = sorted([item.replace(".flac", ".npz") for item in reverb_dir])
    def __len__(self):
        return len(self.clean_files) if not self.debug else 100


    def __getitem__(self, idx):
        clean = np.load(self.clean_files[idx])
        reverb = np.load(self.reverb_files[idx])
        clean = torch.tensor(clean["vae"])
        reverb = torch.tensor(reverb["vae"])
        min_len = min(clean.shape[-1], reverb.shape[-1])
        clean = clean[:,:min_len]
        reverb = reverb[:,:min_len]
        # assert clean.shape == reverb.shape, clean.shape
        clean = clean.transpose(1,0)
        reverb = reverb.transpose(1,0)
        
        return {
            'clean_vae': clean,
            'reverb_vae': reverb,
            'length': clean.shape[0],
            'id': self.clean_files[idx]
        }
    
    def collate(self, batch):
        out = {key:[] for key in batch[0]}
        for item in batch:
            for key, val in item.items():
                out[key].append(val)
                
        out["length"] = torch.LongTensor(out["length"])
        out['clean_vae'] = torch.nn.utils.rnn.pad_sequence(out['clean_vae'], batch_first=True, padding_value=0.0)
        out['reverb_vae'] = torch.nn.utils.rnn.pad_sequence(out['reverb_vae'], batch_first=True, padding_value=0.0)
        return out

    def get_infos(self):
        return self.base_dataset.get_infos()
    



class DistortionEvalDataset(Dataset):
    def __init__(
        self,
        distort_dict: Dict[str, List[Tuple[str, str]]],
        max_items_per_distortion: Optional[int] = None,
        seed: int = 2026,
    ):
        super().__init__()



        # Fix a deterministic mapping name -> id
        self.distortion_names = sorted(list(distort_dict.keys()))
        self.name_to_id = {n: i for i, n in enumerate(self.distortion_names)}

        # Flatten to a list of (dist_id, noisy_path, clean_path)
        items = []
        rng = random.Random(seed)

        for dist_name in self.distortion_names:
            pairs = list(distort_dict[dist_name])

            # optionally subsample for faster analysis
            if max_items_per_distortion is not None and len(pairs) > max_items_per_distortion:
                rng.shuffle(pairs)
                pairs = pairs[:max_items_per_distortion]

            dist_id = self.name_to_id[dist_name]
            for noisy_path, clean_path in pairs:
                items.append((dist_id, dist_name, noisy_path, clean_path))

        self.items = items

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx: int):
        
        dist_id, dist_name, noisy_path, clean_path = self.items[idx]
        clean = np.load(clean_path)
        reverb = np.load(noisy_path)
        clean = torch.tensor(clean["vae"])
        reverb = torch.tensor(reverb["vae"])
        min_len = min(clean.shape[-1], reverb.shape[-1])
        clean = clean[:,:min_len]
        reverb = reverb[:,:min_len]
        # assert clean.shape == reverb.shape, clean.shape
        clean = clean.transpose(1,0)
        reverb = reverb.transpose(1,0)
        

        return {
            "noisy_vae": reverb,                     # [T]
            "clean_vae": clean,                     # [T] or None
            'length': clean.shape[0],                # samples length (or frames later)
            "distortion_id": dist_id,               # int
            "distortion_name": dist_name,           # str
            "noisy_path": noisy_path,
            "clean_path": clean_path,
        }
    

    
    def collate(self, batch):
        out = {key:[] for key in batch[0]}
        for item in batch:
            for key, val in item.items():
                out[key].append(val)
                
        out["length"] = torch.LongTensor(out["length"])
        out['clean_vae'] = torch.nn.utils.rnn.pad_sequence(out['clean_vae'], batch_first=True, padding_value=0.0)
        out['noisy_vae'] = torch.nn.utils.rnn.pad_sequence(out['noisy_vae'], batch_first=True, padding_value=0.0)
        out["distortion_id"] = torch.tensor([b["distortion_id"] for b in batch], dtype=torch.long)
        out["distortion_name"] = [b["distortion_name"] for b in batch]
        out["noisy_path"] = [b["noisy_path"] for b in batch]
        out["clean_path"] = [b["clean_path"] for b in batch]

        return out

    def get_infos(self):
        return self.base_dataset.get_infos()
    







class TSEDataset(Dataset):
    def __init__(
            self, 
            reverb_dir, 
            clean_dir, 
            debug=False,
        ):
        self.debug = debug
        self.clean_files = sorted(glob(os.path.join(clean_dir, '*.npz')))
        self.reverb_files = [item.replace(clean_dir, reverb_dir) for item in self.clean_files]

    def __len__(self):
        return len(self.clean_files) if not self.debug else 100


    def __getitem__(self, idx):
        clean = np.load(self.clean_files[idx])
        reverb = np.load(self.reverb_files[idx])
        clean = torch.tensor(clean["vae"])
        reverb = torch.tensor(reverb["vae"])
        assert clean.shape == reverb.shape, clean.shape
        clean = clean.transpose(1,0)
        reverb = reverb.transpose(1,0)
        
        return {
            'clean_vae': clean,
            'reverb_vae': reverb,
            'length': clean.shape[0],
            'id': self.clean_files[idx]
        }
    
    def collate(self, batch):
        out = {key:[] for key in batch[0]}
        for item in batch:
            for key, val in item.items():
                out[key].append(val)
                
        out["length"] = torch.LongTensor(out["length"])
        out['clean_vae'] = torch.nn.utils.rnn.pad_sequence(out['clean_vae'], batch_first=True, padding_value=0.0)
        out['reverb_vae'] = torch.nn.utils.rnn.pad_sequence(out['reverb_vae'], batch_first=True, padding_value=0.0)
        return out

    def get_infos(self):
        return self.base_dataset.get_infos()

class TSRDataset(Dataset):
    def __init__(
            self, 
            csv_dir, 
            base_dir, 
            vae_dir, 
            task="sep_noisy", 
            sample_rate=16000, 
            vae_rate=50,
            n_src=2, 
            min_length=3, 
            debug=False,
            training=False,
        ):
        self.base_dataset = LibriMix(csv_dir, task, sample_rate, n_src, None)
        self.data_aux = read_enrollment_csv(Path(csv_dir) / 'mixture2enrollment.csv')
        self.seg_len = self.base_dataset.seg_len
        self.data_aux_list = [(m,u) for m in self.data_aux 
                                    for u in self.data_aux[m]]
        self.debug = debug
        self.sample_rate = sample_rate
        self.base_dir = base_dir
        self.vae_dir = vae_dir
        self.vae_rate = vae_rate
        self.min_length = int(min_length*vae_rate)
        self.training = training
        
    def __len__(self):
        return len(self.data_aux_list) if not self.debug else len(self.data_aux_list) // 400


    def __getitem__(self, idx):
        mix_id, utt_id = self.data_aux_list[idx]
        row = self.base_dataset.df[self.base_dataset.df['mixture_ID'] == mix_id].squeeze()
        mixture_path = row['mixture_path']
        tgt_spk_idx = mix_id.split('_').index(utt_id)

        # read mixture
        mixture = torch.load(mixture_path.replace(self.base_dir, self.vae_dir).replace('.wav','.pt'))
        source_path = row[f'source_{tgt_spk_idx+1}_path']
        source = torch.load(source_path.replace(self.base_dir, self.vae_dir).replace('.wav','.pt'))
        exclude_path = source_path.replace('/s1/', '/s2/') if '/s1/' in source_path else source_path.replace('/s2/', '/s1/')
        exclude = torch.load(exclude_path.replace(self.base_dir, self.vae_dir).replace('.wav','.pt'))
        assert mixture.shape == source.shape, mixture.shape
        assert source.shape == exclude.shape, exclude.shape
        mixture = mixture.transpose(1,0)
        source = source.transpose(1,0)
        exclude = exclude.transpose(1,0)

        if self.training:
            if mixture.shape[0] > self.min_length:
                new_length = random.randint(self.min_length, mixture.shape[0])
                start = random.randint(0, mixture.shape[0]-new_length)
                mixture = mixture[start:start+new_length]
                source = source[start:start+new_length]
                exclude = exclude[start:start+new_length]

            
        return {
            'mixture_vae': mixture,
            'source_vae': source,
            'exclude_vae': exclude,
            'length': mixture.shape[0],
            'id': mix_id,
            'mixture_path': mixture_path,
            'source_path': source_path,
            'exclude_path': exclude_path
        }
    
    
    
    def collate(self, batch):
        out = {key:[] for key in batch[0]}
        for item in batch:
            for key, val in item.items():
                out[key].append(val)
                
        out["length"] = torch.LongTensor(out["length"])
        out['mixture_vae'] = torch.nn.utils.rnn.pad_sequence(out['mixture_vae'], batch_first=True, padding_value=0.0)
        out['source_vae'] = torch.nn.utils.rnn.pad_sequence(out['source_vae'], batch_first=True, padding_value=0.0)
        out['exclude_vae'] = torch.nn.utils.rnn.pad_sequence(out['exclude_vae'], batch_first=True, padding_value=0.0)
        return out

    def get_infos(self):
        return self.base_dataset.get_infos()

if __name__ == "__main__":
    pass