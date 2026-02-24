# Quick Use

This page provides a quick-start guide for using them. The environment was built based on Solospeech as we reused the (de)compressor.


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

## training

The following lines are examples for training. 

```bash
cd ditflow/solospeech/scripts/solospeech
export PYTHONPATH=$PWD:$PYTHONPATH
accelerate launch scripts/solospeech/train-tse-cfm.py --train-clean <your_clean_files_dir> --train-reverb <your_distorted_files_dir>
```


You can select file based on finetuning method you want, LoRA or Mix of LoRA Experts.

```bash
accelerate launch scripts/solospeech/train-tse-cfm_urgent2026_MoLExLoRA_lb_orth_attn.py 
```

Use the following to control the hyperparameters for finetuning:
--use-molex --molex-lb-type switch --molex-lb-coef 1e-2 --molex-orth-coef 1.0


## inference

The following lines are examples for inference. 

```bash
cd ditflow/solospeech/scripts/solospeech
export PYTHONPATH=$PWD:$PYTHONPATH
accelerate launch scripts/solospeech/batch_test_cfm.py --input_dir <your_input_files_dir> --save-dir <your_saved_files_dir>
```