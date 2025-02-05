#!/bin/bash
#SBATCH --job-name=bird
#SBATCH --mem=64G
#SBATCH --cpus-per-task=16
#SBATCH --time=24:00:00
#SBATCH --partition=gpua100
#SBATCH --gres=gpu:1
#SBATCH --output=logs/o-%j.log
#SBATCH --error=logs/e-%j.log

# Load the conda environment
source /gpfs/users/yutaoc/miniconda3/etc/profile.d/conda.sh
conda activate bird 

torchrun --nproc_per_node=4 train.py \
    --model eva02_large_patch14_448.mim_m38m_ft_in22k_in1k --epochs 200 --batch-size 32 --opt adamw --lr 0.001 --wd 0.0001 \
    --lr-scheduler cosineannealinglr --lr-warmup-method linear --lr-warmup-epochs 10 \
    --lr-warmup-decay 0.033 --lr-warmup-method linear --amp --label-smoothing 0.11 --mixup-alpha 0.4 \
    --auto-augment ta_wide --output-dir /gpfs/workdir/yutaoc/bird/checkpoints/eva02 --random-erase 0.3 \
    --train-crop-size 448 --val-crop-size 448 --val-resize-size 448 --use-v2 \
    --clip-grad-norm 1 --ra-sampler --cutmix-alpha 0.3 --model-ema --opt adamw --interpolation bicubic \

torchrun --nproc_per_node=4 train.py \
    --model eva02_large_patch14_448.mim_m38m_ft_in22k_in1k --epochs 200 --batch-size 8 --opt adamw --lr 0.00001 --wd 0.00001 \
    --lr-scheduler cosineannealinglr --lr-warmup-method linear --lr-warmup-epochs 10 \
    --lr-warmup-decay 0.033 --lr-warmup-method linear --amp --label-smoothing 0.1 --mixup-alpha 0.3 \
    --auto-augment ta_wide --output-dir /gpfs/workdir/yutaoc/bird/checkpoints/eva02/ft --random-erase 0.3 \
    --checkpoint-dir /gpfs/workdir/yutaoc/bird/checkpoints/eva02-check --train-crop-size 448 --val-crop-size 448 \
    --val-resize-size 448 --use-v2 --ft-backbone --clip-grad-norm 1 --ra-sampler --cutmix-alpha 0.3 \
    --model-ema --opt adamw --interpolation bicubic

# python3 test.py --checkpoint_dir /gpfs/workdir/yutaoc/bird/checkpoints/eva02/ft --model eva02_large_patch14_448.mim_m38m_ft_in22k_in1k