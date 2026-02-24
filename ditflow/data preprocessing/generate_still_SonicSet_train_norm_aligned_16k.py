import numpy as np
import librosa
import os
import matplotlib.pyplot as plt
from IPython.display import Audio
from scipy import signal
import glob
import IPython.display as ipd
import soundfile as sf
import scipy.stats as stats
# import speechmetrics
import scipy

from tqdm import tqdm
import pandas as pd
from pydub import AudioSegment

import torch
import json
from collections import defaultdict

import pyloudnorm as pyln

import math

import torch
import librosa
import torchaudio
from scipy import signal
from torchaudio.transforms import Resample

import random

import shutil



random.seed(218)  # Set seed for reproducibility (218, 223, 224)




# def normalize(audio, norm='peak'):
#     if norm == 'peak':
#         peak = abs(audio).max()
#         if peak != 0:
#             return audio / peak
#         else:
#             return audio
#     elif norm == 'rms':
#         if torch.is_tensor(audio):
#             audio = audio.numpy()
#         audio_without_padding = np.trim_zeros(audio, trim='b')
#         rms = np.sqrt(np.mean(np.square(audio_without_padding))) * 100
#         if rms != 0:
#             return audio / rms
#         else:
#             return audio
#     else:
#         raise NotImplementedError






# Initialize a dictionary to hold the structure
speaker_data = {}

# Define the root directory containing speakerID folders
root_dir = "/export/corpora5/LibriSpeech/train-clean-360"

# Walk through the directory structure
for speakerID in os.listdir(root_dir):
    speaker_path = os.path.join(root_dir, speakerID)
    if os.path.isdir(speaker_path):  # Ensure it's a directory
        speaker_data[speakerID] = {}

        # Walk through each ChapterID folder inside the speakerID directory
        for chapterID in os.listdir(speaker_path):
            chapter_path = os.path.join(speaker_path, chapterID)
            if os.path.isdir(chapter_path):  # Ensure it's a directory
                # Initialize a list to store utterances for the current chapter
                speaker_data[speakerID][chapterID] = []

                # Collect the speech utterances for the current ChapterID
                for utterance in os.listdir(chapter_path):
                    utterance_path = os.path.join(chapter_path, utterance)
                    if os.path.isfile(utterance_path) and utterance_path[-4:] == "flac":  # Ensure it's a file (utterance)
                        speaker_data[speakerID][chapterID].append(utterance_path)


# Example list of paths
RIR_pt_file_paths = glob.glob("/export/fs05/tcao7/SonicSim/SonicSet_v2/home/likai/data/SonicSim/SonicSet/train/*/*/*.pt")

# Dictionary to store room-wise classification
room_dict = defaultdict(list)

# Process each path
for path in RIR_pt_file_paths:
    parts = path.strip("/").split("/")  # Split path by "/"
    
    roomID = parts[-3]  # Extract roomID
    room_dict[roomID].append(path)  # Store path under roomID



# Flatten the dictionary into a list of tuples (roomID, file_path)
RIR_pt_paths_tuplelist = [(roomID, file_path) for roomID, file_paths in room_dict.items() for file_path in file_paths]



# Step 1: Flatten the dictionary into a list of tuples (speakerID, utterance_path)
utterance_list = []
for speakerID, chapters in speaker_data.items():
    for chapterID, utterances in chapters.items():
        for utterance in utterances:
            utterance_list.append((speakerID, utterance))

# Step 2: Shuffle the utterances randomly
random.shuffle(utterance_list)

# Step 3: Group into sets of three different speakers
selected_utterances = []
used_speakers = set()

while utterance_list:
    batch = []
    to_remove = []  # Track selected items to remove after iteration

    for item in utterance_list:
        speaker, utterance = item
        if speaker not in used_speakers:
            batch.append(item)
            used_speakers.add(speaker)
            to_remove.append(item)  # Mark item for removal

        if len(batch) == 3:  # Stop once we get three unique speakers
            break

    # Remove selected items from utterance_list
    for item in to_remove:
        utterance_list.remove(item)

    selected_utterances.append(batch)
    used_speakers = set()  # Reset used speakers for the next batch


# Step 4: Handle remaining utterances (if any)
if utterance_list:  # If we have 1 or 2 leftover utterances, add them as the last batch
    selected_utterances.append(utterance_list)


for selected_utterance in tqdm(selected_utterances):
    # selected_utterance e.g., [('7176',
    #    '/export/corpora5/LibriSpeech/test-clean/7176/92135/7176-92135-0040.flac'),
    #   ('2300',
    #    '/export/corpora5/LibriSpeech/test-clean/2300/131720/2300-131720-0039.flac'),
    #   ('4970',
    #    '/export/corpora5/LibriSpeech/test-clean/4970/29093/4970-29093-0012.flac')]

    # Randomly select one RIR pt file path
    random_roomID, random_file = random.choice(RIR_pt_paths_tuplelist) 
    # e.g., ('gTV8FGcVJC9','/export/fs05/tcao7/SonicSim/SonicSet_v2/home/likai/data/SonicSim/SonicSet/test/gTV8FGcVJC9/2094-260-1221/rir_save_test_Mono.pt')

    files = torch.load(random_file)
    parent_path = os.path.dirname(random_file)


    if not os.path.exists("/export/fs05/tcao7/stillSonicSetNormAligned16k/train/"+random_roomID):
        os.makedirs("/export/fs05/tcao7/stillSonicSetNormAligned16k/train/"+random_roomID)

    # Step 1: Flatten the list into a list of individual 4D tensors
    all_tensors = [t[i] for t in files for i in range(t.shape[0])]

    # Step 2: Randomly select three tensor from the flattened list for three speaker/different positions
    selected_tensor = random.sample(all_tensors, k=3)


    joined_ids =  "-".join(item[0] for item in selected_utterance)
    save_path = "/export/fs05/tcao7/stillSonicSetNormAligned16k/train/"+ random_roomID + "/" + joined_ids
        
    if not os.path.exists(save_path):
        os.makedirs(save_path)


    reverb_wavs =[]
    filenames = []
    RIRs_save = []
    for speech_path_idx, speech_ID_path_tuple in enumerate(selected_utterance):
        # first_elements_speaker_ID = [item[0] for item in selected_utterance]
        
        


        waveform, sr = torchaudio.load(speech_ID_path_tuple[1])

        # Define the new sample rate
        # new_sample_rate = 8000

        # # Resample the audio
        # resampler = Resample(orig_freq=sr, new_freq=new_sample_rate)
        # resampled_waveform = resampler(waveform)

        waveform = waveform.numpy()

        filenames.append(speech_ID_path_tuple[1].split("/")[-1]) # e.g., '5142-36377-0004.flac'


        # print(waveform.reshape(1, -1).shape)
        # print(selected_tensor[speech_path_idx][0][0].numpy().shape)
        reverb_wav = signal.fftconvolve(waveform.reshape(1, -1), selected_tensor[speech_path_idx][0][0].numpy().reshape(1, -1), mode="full")[:, : waveform.shape[-1]]
        
        # alignment

        corr = np.correlate(waveform.flatten(), reverb_wav.flatten(), mode='full')
        lags = np.arange(-waveform.shape[1] + 1, waveform.shape[1])
        lag_max = lags[np.argmax(corr)]

        direct_idx = abs(lag_max)
        reverb_wav_aligned = signal.fftconvolve(waveform.reshape(1, -1), selected_tensor[speech_path_idx][0][0].numpy().reshape(1, -1), mode="full")[:, direct_idx: direct_idx + waveform.shape[-1]]
        reverb_wavs.append(reverb_wav_aligned)

        RIRs_save.append(selected_tensor[speech_path_idx][0][0].numpy().reshape(1, -1))




    # Concatenate along dimension 1
    combined_waveform = np.concatenate(reverb_wavs, axis=1)  # Shape: [1, total_length]

    # Compute the maximum absolute value and normalize
    max_abs_value = np.max(np.abs(combined_waveform))
    
    # peak = abs(audio).max()
    if max_abs_value != 0:
        normalized_waveforms = [waveform / max_abs_value for waveform in reverb_wavs]
    else:
        normalized_waveforms = reverb_wavs


    for save_idx in range(len(filenames)):
        torchaudio.save(save_path + "/" + filenames[save_idx], torch.from_numpy(reverb_wavs[save_idx]), sr)
        np.save(save_path + "/" + filenames[save_idx].split(".")[0] +'.npy', RIRs_save[save_idx])

    # torchaudio.save(save_path + "/" + filenames[1], torch.from_numpy(reverb_wavs[1]), new_sample_rate)
    # torchaudio.save(save_path + "/" + filenames[2], torch.from_numpy(reverb_wavs[2]), new_sample_rate)

    # resample music and noise
    music_audio, music_sr = torchaudio.load(parent_path+"/music_audio.wav")
    # resampled_music = resampler(music_audio)
    torchaudio.save(save_path + "/music_audio.wav", music_audio, music_sr)

    noise_audio, noise_sr = torchaudio.load(parent_path+"/noise_audio.wav")
    # resampled_noise = resampler(noise_audio)
    torchaudio.save(save_path + "/noise_audio.wav", noise_audio, noise_sr)


    shutil.copy(parent_path+"/trace.png", save_path + "/trace.png") 
    shutil.copy(parent_path+"/rir_save_train_Mono.pt", save_path + "/rir_save_train_Mono.pt")
        
























