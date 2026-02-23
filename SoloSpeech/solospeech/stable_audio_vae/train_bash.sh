export CUDA_VISIBLE_DEVICES=0

python train.py --dataset-config './configs/vae_data.txt' --model-config './configs/stftvae_16k.config' --name 'stftvae_16k_base'