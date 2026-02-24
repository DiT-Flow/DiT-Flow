# Quick Use

This page provides a quick-start guide for generating RIRs and convolute them with your speech dataset. In our paper, we use LibriSpeech as our base dataset. Make sure you download SonicSet dataset and set the correct the path "root_dir" in our script.

The following lines are examples for training. 

```bash
cd ditflow/data preprocessing
python generate_still_SonicSet_val_norm_aligned_16k.py
```