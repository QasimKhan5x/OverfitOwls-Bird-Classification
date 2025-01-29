#!/bin/bash
#SBATCH --job-name=bird
#SBATCH --mem=32G
#SBATCH --cpus-per-task=16
#SBATCH --time=24:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:4
#SBATCH --output=logs/o-%j.log
#SBATCH --error=logs/e-%j.log

# Load the conda environment
source /gpfs/users/yutaoc/miniconda3/etc/profile.d/conda.sh
conda activate bird 

# torchrun --nproc_per_node=4 train.py \
#     --model vit_l_32 --epochs 300 --batch-size 32 --opt adamw --lr 0.001 --wd 0.0001 \
#     --lr-scheduler cosineannealinglr --lr-warmup-method linear --lr-warmup-epochs 10 \
#     --lr-warmup-decay 0.033 --lr-warmup-method linear --amp --label-smoothing 0.1 --mixup-alpha 0.2 \
#     --auto-augment ra --output-dir checkpoints/dino_combined --random-erase 0.1 \
#     --train-crop-size 392 --val-crop-size 392 --val-resize-size 392 --use-v2 \
#     --clip-grad-norm 1 --ra-sampler --cutmix-alpha 1.0 --model-ema --opt adamw

# torchrun --nproc_per_node=4 train.py \
#     --model vit_l_32 --epochs 100 --batch-size 32 --opt adamw --lr 0.0001 --wd 0.0001 \
#     --lr-scheduler cosineannealinglr --lr-warmup-method linear --lr-warmup-epochs 10 \
#     --lr-warmup-decay 0.033 --lr-warmup-method linear --amp --label-smoothing 0.1 --mixup-alpha 0.2 \
#     --auto-augment ra --output-dir checkpoints/3 --random-erase 0.0 \
#     --train-crop-size 392 --val-crop-size 392 --val-resize-size 392 --use-v2 \
#     --clip-grad-norm 1 --ra-sampler --cutmix-alpha 1.0 --model-ema --opt adamw

python3 test.py --checkpoint_dir checkpoints/dino_combined