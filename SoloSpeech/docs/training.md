# Training


## Audio Compressor Training

To train a T-F Audio VAE model, please:

1. Change data path in `capspeech/stable_audio_vae/configs/vae_data.txt` (any folder contains audio files).

2. Change model config in `capspeech/stable_audio_vae/configs/stftvae_16k_320x.config`. 

We provide config for training audio files of 16k sampling rate,  please change the settings when you want other sampling rates.

3. Change batch size and training settings in `capspeech/stable_audio_vae/defaults.ini`.

4. Run:

```bash
cd capspeech/stable_audio_vae/
bash train_bash.sh
``` 

## Target Extractor Training

To train a targer extractor, please:

1. Prepare audio files following [SpeakerBeam](https://github.com/BUTSpeechFIT/speakerbeam).

2. Prepare latent features:

```bash
python capspeech/dataset/extract_vae.py
```

3. Training:
```bash
accelerate launch capspeech/scripts/solospeech/train-tse.py
```

## Corrector Training

To train a corrector, please run:
```bash
CUDA_VISIBLE_DEVICES=0 python capspeech/corrector/train-fastgeco.py --gpus 1 --batch_size 16
```