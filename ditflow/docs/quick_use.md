# Quick Use

We release our best models for target speech extraction. This page provides a quick-start guide for using them.


## Install

For Linux developers and researchers, run:

```bash
conda create -n solospeech python=3.8.19
conda activate solospeech
git clone https://github.com/WangHelin1997/SoloSpeech
cd SoloSpeech/
pip install -r requirements.txt
pip install .
```

## Target Speech Extraction

This is an example to run SoloSpeech:
```bash
python scripts/test.py \
    --test-wav "../assets/test2.wav"
    --enroll-wav "../assets/test2_enroll.wav"
    --output-path "./demo/test2_solospeech.wav"
```
Here, `--test-wav` is the path of the mixture audio, `--enroll-wav` is the path of the enrollment audio (representing the speaker you want to extract), and `--output-path` is the path to save output audio.

