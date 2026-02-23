from os.path import join 
from glob import glob
from argparse import ArgumentParser
from soundfile import read
from tqdm import tqdm
from pesq import pesq
import pandas as pd
import librosa
from pystoi import stoi
import numpy as np
import os
import argparse
import numpy.polynomial.polynomial as poly
import onnxruntime as ort
import soundfile as sf
from jiwer import wer as calculate_wer
from jiwer import cer as calculate_cer
from transformers import Wav2Vec2FeatureExtractor, WavLMForXVector
from whisper.normalizers import EnglishTextNormalizer
import whisper
import torch
import torchaudio

SAMPLING_RATE = 16000
INPUT_LENGTH = 9.01
normalizer = EnglishTextNormalizer()
device = "cuda" if torch.cuda.is_available() else "cpu"
whisper_model = whisper.load_model("large-v3-turbo", device=device)
xvector_extractor = Wav2Vec2FeatureExtractor.from_pretrained('microsoft/wavlm-base-plus-sv')
xvector_computer = WavLMForXVector.from_pretrained('microsoft/wavlm-base-plus-sv')
cosine_sim = torch.nn.CosineSimilarity(dim=-1)

def asr(wav_path):
    result = whisper_model.transcribe(wav_path)
    pred = result['text'].strip()
    pred = normalizer(pred)
    return pred

class ComputeScore:
    def __init__(self, primary_model_path, p808_model_path) -> None:
        self.onnx_sess = ort.InferenceSession(primary_model_path)
        self.p808_onnx_sess = ort.InferenceSession(p808_model_path)
        
    def audio_melspec(self, audio, n_mels=120, frame_size=320, hop_length=160, sr=16000, to_db=True):
        mel_spec = librosa.feature.melspectrogram(y=audio, sr=sr, n_fft=frame_size+1, hop_length=hop_length, n_mels=n_mels)
        if to_db:
            mel_spec = (librosa.power_to_db(mel_spec, ref=np.max)+40)/40
        return mel_spec.T

    def get_polyfit_val(self, sig, bak, ovr, is_personalized_MOS):
        if is_personalized_MOS:
            p_ovr = np.poly1d([-0.00533021,  0.005101  ,  1.18058466, -0.11236046])
            p_sig = np.poly1d([-0.01019296,  0.02751166,  1.19576786, -0.24348726])
            p_bak = np.poly1d([-0.04976499,  0.44276479, -0.1644611 ,  0.96883132])
        else:
            p_ovr = np.poly1d([-0.06766283,  1.11546468,  0.04602535])
            p_sig = np.poly1d([-0.08397278,  1.22083953,  0.0052439 ])
            p_bak = np.poly1d([-0.13166888,  1.60915514, -0.39604546])

        sig_poly = p_sig(sig)
        bak_poly = p_bak(bak)
        ovr_poly = p_ovr(ovr)

        return sig_poly, bak_poly, ovr_poly

    def __call__(self, fpath, sampling_rate, is_personalized_MOS):
        aud, input_fs = sf.read(fpath)
        fs = sampling_rate
        if input_fs != fs:
            audio = librosa.resample(aud, input_fs, fs)
        else:
            audio = aud
        actual_audio_len = len(audio)
        len_samples = int(INPUT_LENGTH*fs)
        while len(audio) < len_samples:
            audio = np.append(audio, audio)
        
        num_hops = int(np.floor(len(audio)/fs) - INPUT_LENGTH)+1
        hop_len_samples = fs
        predicted_mos_sig_seg_raw = []
        predicted_mos_bak_seg_raw = []
        predicted_mos_ovr_seg_raw = []
        predicted_mos_sig_seg = []
        predicted_mos_bak_seg = []
        predicted_mos_ovr_seg = []
        predicted_p808_mos = []

        for idx in range(num_hops):
            audio_seg = audio[int(idx*hop_len_samples) : int((idx+INPUT_LENGTH)*hop_len_samples)]
            if len(audio_seg) < len_samples:
                continue

            input_features = np.array(audio_seg).astype('float32')[np.newaxis,:]
            p808_input_features = np.array(self.audio_melspec(audio=audio_seg[:-160])).astype('float32')[np.newaxis, :, :]
            oi = {'input_1': input_features}
            p808_oi = {'input_1': p808_input_features}
            p808_mos = self.p808_onnx_sess.run(None, p808_oi)[0][0][0]
            mos_sig_raw,mos_bak_raw,mos_ovr_raw = self.onnx_sess.run(None, oi)[0][0]
            mos_sig,mos_bak,mos_ovr = self.get_polyfit_val(mos_sig_raw,mos_bak_raw,mos_ovr_raw,is_personalized_MOS)
            predicted_mos_sig_seg_raw.append(mos_sig_raw)
            predicted_mos_bak_seg_raw.append(mos_bak_raw)
            predicted_mos_ovr_seg_raw.append(mos_ovr_raw)
            predicted_mos_sig_seg.append(mos_sig)
            predicted_mos_bak_seg.append(mos_bak)
            predicted_mos_ovr_seg.append(mos_ovr)
            predicted_p808_mos.append(p808_mos)

        clip_dict = {'filename': fpath, 'len_in_sec': actual_audio_len/fs, 'sr':fs}
        clip_dict['num_hops'] = num_hops
        clip_dict['OVRL_raw'] = np.mean(predicted_mos_ovr_seg_raw)
        clip_dict['SIG_raw'] = np.mean(predicted_mos_sig_seg_raw)
        clip_dict['BAK_raw'] = np.mean(predicted_mos_bak_seg_raw)
        clip_dict['OVRL'] = np.mean(predicted_mos_ovr_seg)
        clip_dict['SIG'] = np.mean(predicted_mos_sig_seg)
        clip_dict['BAK'] = np.mean(predicted_mos_bak_seg)
        clip_dict['P808_MOS'] = np.mean(predicted_p808_mos)
        return clip_dict

def mean_std(data):
    data = data[~np.isnan(data)]
    mean = np.mean(data)
    std = np.std(data)
    return mean, std

def si_sdr(estimate, reference):
    """
    Compute Scale-Invariant Signal-to-Distortion Ratio (SI-SDR) using numpy.

    Args:
        estimate (np.ndarray): Estimated signal of shape (num_samples,) or (batch_size, num_samples)
        reference (np.ndarray): Reference (clean) signal of the same shape as estimate.
    
    Returns:
        np.ndarray: SI-SDR value for each signal in the batch (or a single value if single sample).
    """
    # Ensure both inputs are of the same shape
    assert estimate.shape == reference.shape, "Estimate and reference must have the same shape"
    
    # If the inputs are 1D, we convert them to 2D arrays with shape (1, num_samples)
    if estimate.ndim == 1:
        estimate = estimate[np.newaxis, :]
        reference = reference[np.newaxis, :]

    # Remove the mean (DC component) from both signals
    estimate = estimate - np.mean(estimate, axis=-1, keepdims=True)
    reference = reference - np.mean(reference, axis=-1, keepdims=True)
    
    # Compute the scale factor (projection of estimate onto reference)
    scale = np.sum(reference * estimate, axis=-1, keepdims=True) / np.sum(reference ** 2, axis=-1, keepdims=True)
    
    # Compute the true (scaled) reference signal
    projection = scale * reference
    
    # Compute the error (distortion)
    noise = estimate - projection
    
    # Compute SI-SDR
    sdr = 10 * np.log10(np.sum(projection ** 2, axis=-1) / np.sum(noise ** 2, axis=-1))
    
    return sdr if len(sdr) > 1 else sdr[0]
    

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--test_dir", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default=None)
    args = parser.parse_args()

    data = {"filename": [], "pesq": [], "estoi": [], "sisdr": [], "OVRL": [], "SIG": [], "BAK": [], "P808_MOS": [], "wer": [], "cer":[], "sim": []}

    # Evaluate standard metrics
    noisy_files = sorted(glob(join(args.test_dir, '*_pred.wav')))
    clean_files = [item.replace('_pred.wav', '_ref.wav') for item in noisy_files]
    p808_model_path = os.path.join('DNSMOS', 'model_v8.onnx')
    primary_model_path = os.path.join('DNSMOS', 'sig_bak_ovr.onnx')
    dnsmodel = ComputeScore(primary_model_path, p808_model_path)
    
    # noisy_files = noisy_files[:10]
    # clean_files = clean_files[:10]
    
    for noisy_file, clean_file in tqdm(zip(noisy_files, clean_files)):
        filename = clean_file.split('/')[-1]
        x, _ = librosa.load(clean_file,sr=16000,mono=True)
        x = x * 0.9 / max(abs(x))
        x_hat, _ = librosa.load(noisy_file, sr=16000,mono=True)
        x_hat = x_hat * 0.9 / max(abs(x_hat))
        leng = min(len(x_hat),len(x))
        x = x[:leng]
        x_hat = x_hat[:leng]
        dnsmos = dnsmodel(noisy_file, 16000, False)
        # asr
        gt_wav, sr = torchaudio.load(clean_file)
        if sr != 16000:
            resampler = transforms.Resample(orig_freq=sr, new_freq=16000)
            gt_wav = resampler(gt_wav)
        pred_wav, sr = torchaudio.load(noisy_file)
        if sr != 16000:
            resampler = transforms.Resample(orig_freq=sr, new_freq=16000)
            pred_wav = resampler(pred_wav)
        gt_asr = asr(clean_file)
        pred_asr = asr(noisy_file)
        wer = round(calculate_wer(gt_asr, pred_asr), 3)
        cer = round(calculate_cer(gt_asr, pred_asr), 3)
        # speaker sim
        audio = [gt_wav[0].cpu().numpy(), pred_wav[0].cpu().numpy()]
        inputs = xvector_extractor(audio, sampling_rate=16000, padding=True, return_tensors="pt")
        embeddings = xvector_computer(**inputs).embeddings
        embeddings = torch.nn.functional.normalize(embeddings, dim=-1).cpu()
        sim = cosine_sim(embeddings[0], embeddings[1]).item()

        data["filename"].append(filename)
        data["pesq"].append(pesq(16000, x, x_hat, 'wb'))
        data["estoi"].append(stoi(x, x_hat, 16000, extended=True))
        data["sisdr"].append(si_sdr(x_hat, x))
        data["OVRL"].append(dnsmos["OVRL"])
        data["SIG"].append(dnsmos["SIG"])
        data["BAK"].append(dnsmos["BAK"])
        data["P808_MOS"].append(dnsmos["P808_MOS"])
        data["wer"].append(wer)
        data["cer"].append(cer)
        data["sim"].append(sim)

    # Save results as DataFrame    
    df = pd.DataFrame(data)

    # Print results
    print("PESQ: {:.3f} ± {:.3f}".format(*mean_std(df["pesq"].to_numpy())))
    print("ESTOI: {:.3f} ± {:.3f}".format(*mean_std(df["estoi"].to_numpy())))
    print("SISDR: {:.3f} ± {:.3f}".format(*mean_std(df["sisdr"].to_numpy())))
    print("OVRL: {:.3f} ± {:.3f}".format(*mean_std(df["OVRL"].to_numpy())))
    print("SIG: {:.3f} ± {:.3f}".format(*mean_std(df["SIG"].to_numpy())))
    print("BAK: {:.3f} ± {:.3f}".format(*mean_std(df["BAK"].to_numpy())))
    print("P808_MOS: {:.3f} ± {:.3f}".format(*mean_std(df["P808_MOS"].to_numpy())))
    print("WER: {:.3f} ± {:.3f}".format(*mean_std(df["wer"].to_numpy())))
    print("CER: {:.3f} ± {:.3f}".format(*mean_std(df["cer"].to_numpy())))
    print("SIM: {:.3f} ± {:.3f}".format(*mean_std(df["sim"].to_numpy())))

    os.makedirs(args.output_dir, exist_ok=True)
    # Save average results to file
    log = open(join(args.output_dir, "_avg_results.txt"), "w")
    log.write("PESQ: {:.3f} ± {:.3f}".format(*mean_std(df["pesq"].to_numpy())) + "\n")
    log.write("ESTOI: {:.3f} ± {:.3f}".format(*mean_std(df["estoi"].to_numpy())) + "\n")
    log.write("SISDR: {:.3f} ± {:.3f}".format(*mean_std(df["sisdr"].to_numpy())) + "\n")
    log.write("OVRL: {:.3f} ± {:.3f}".format(*mean_std(df["OVRL"].to_numpy())) + "\n")
    log.write("SIG: {:.3f} ± {:.3f}".format(*mean_std(df["SIG"].to_numpy())) + "\n")
    log.write("BAK: {:.3f} ± {:.3f}".format(*mean_std(df["BAK"].to_numpy())) + "\n")
    log.write("P808_MOS: {:.3f} ± {:.3f}".format(*mean_std(df["P808_MOS"].to_numpy())) + "\n")
    log.write("WER: {:.3f} ± {:.3f}".format(*mean_std(df["wer"].to_numpy())) + "\n")
    log.write("CER: {:.3f} ± {:.3f}".format(*mean_std(df["cer"].to_numpy())) + "\n")
    log.write("SIM: {:.3f} ± {:.3f}".format(*mean_std(df["sim"].to_numpy())) + "\n")

    # Save DataFrame as csv file
    df.to_csv(join(args.output_dir, "_results.csv"), index=False)
