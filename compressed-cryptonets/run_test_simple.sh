#!/usr/bin/env bash

# To run in the background:
# nohup bash run_train_simple.sh > {output file location} &

set -e
eval "$(conda shell.bash hook)"
conda activate PT-MAP-sf

# ------ User Arguments ------
gpu=2   # Multi-GPU training is not supported with QAT Brevitas due to quantized BatchNorm sync issues (causing validation loss stagnant)
dataset=ImageNet
dataset_path=/local/a/imagenet/imagenet2012
checkpoint_path=/home/nano01/a/roy208/compressed-cryptonets/checkpoints/ImageNet/ResNet18qat/input_3_224_224_bitwidth_4_finetune/best.tar
num_classes=1000
model=ResNet18qat
test_batch_size=512
dct_status=N
image_size=224
channels=3
filter_size=8
dct_pattern=default
bit_width=4
# ----------------------------
echo "dataset=${dataset}"
echo "num_classes=${num_classes}"
echo "model=${model}"
echo "image_size=${image_size}"
echo "test_batch_size=${test_batch_size}"
echo "dataset_path=${dataset_path}"
echo "checkpoint_path=${checkpoint_path}"
echo "bit_width=${bit_width}"
echo "dct_status=${dct_status}"


if [ "${dct_status}" == Y ]; then
  echo "filter_size=${filter_size}"
  echo "dct_pattern=${dct_pattern}"
  echo -e "channels=${channels}\n"
  CUDA_VISIBLE_DEVICES="${gpu}" python -u test_simple.py \
    --dataset "${dataset}" \
    --dataset_path "${dataset_path}" \
    --checkpoint_path "${checkpoint_path}" \
    --num_classes "${num_classes}" \
    --model "${model}" \
    --image_size_dct "${image_size}" \
    --channels "${channels}" \
    --filter_size "${filter_size}" \
    --test_batch_size "${test_batch_size}" \
    --bit_width "${bit_width}" \
    --dct_pattern "${dct_pattern}" \
    --dct_status
else
  echo -e "\n"
  CUDA_VISIBLE_DEVICES="${gpu}" python -u test_simple.py \
    --dataset "${dataset}" \
    --dataset_path "${dataset_path}" \
    --checkpoint_path "${checkpoint_path}" \
    --num_classes "${num_classes}" \
    --model "${model}" \
    --image_size "${image_size}" \
    --channels "${channels}" \
    --test_batch_size "${test_batch_size}" \
    --bit_width "${bit_width}"
fi