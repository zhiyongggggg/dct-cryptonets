#!/usr/bin/env bash

# To run in the background:
# nohup bash run_train_imagenet.sh > {output file location} &

set -e
eval "$(conda shell.bash hook)"
conda activate PT-MAP-sf

# ------ User Arguments ------
gpu=2,3
# ----------------------------

CUDA_VISIBLE_DEVICES="${gpu}" python -u train_imagenet.py \
  -j 8 \
  --epochs 1 \
  --train-batch 1024 \
  --test-batch 512 \
  --arch ResNet18 \
  --pretrained False \
  -d /local/a/imagenet/imagenet2012/ \
  --checkpoint ~/Repos/compressed-cryptonets/checkpoints/ \
  --gpu-id "${gpu}"
#  --schedule 32 61 \
#  --gamma 0.1 \
#  --subset 24 \
#  --pattern triangle \
#  --resume ~/Repos/compressed-cryptonets/checkpoints/model_best.pth.tar
