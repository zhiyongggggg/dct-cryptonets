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
  -j 16 \
  --test-batch 200 \
  --arch ResNetDCT_Upscaled_Static \
  --pretrained False \
  -d /local/a/imagenet/imagenet2012/ \
  --resume ~/Repos/compressed-cryptonets/checkpoints/resnet50dct_upscaled_static_24_pretrained.pth.tar \
  --subset 24 \
  --pattern square \
  --evaluate