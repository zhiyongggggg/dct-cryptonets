#!/usr/bin/env bash

# To run in the background:
# nohup bash run_train_one_shot.sh > {output file location} &

set -e
eval "$(conda shell.bash hook)"
conda activate PT-MAP-sf

# ------ User Arguments ------
gpu=0
# ----------------------------

## Train non-DCT models
#CUDA_VISIBLE_DEVICES="${gpu}" python -u train_one_shot.py \
#  --dataset miniImagenet \
#  --method baseline++ \
#  --model ResNet18qat \
#  --train_aug \
#  --image_size 224 \
#  --batch_size 210 \
#  --test_batch_size 210 \
#  --stop_epoch 70

# Train DCT models
CUDA_VISIBLE_DEVICES="${gpu}" python -u train_one_shot.py \
  --dataset miniImagenet \
  --method baseline++ \
  --model ResNet10qat \
  --dct_status \
  --train_aug \
  --batch_size 128 \
  --test_batch_size 128 \
  --stop_epoch 5 \
  --image_size_dct 56 \
  --channels 6

