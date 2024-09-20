#!/usr/bin/env bash

# To run in the background:
# nohup bash run_train_simple.sh > {output file location} &

set -e
eval "$(conda shell.bash hook)"
conda activate PT-MAP-sf
export BREVITAS_IGNORE_MISSING_KEYS=1

# ------ User Arguments ------
gpu=3   # Multi-GPU training is not supported with QAT Brevitas due to quantized BatchNorm sync issues (causing validation loss stagnant)
model=ResNet18qat
dataset=miniImagenet
dataset_path=/home/min/a/roy208/datasets/miniImagenet/
checkpoint_dir=/home/nano01/a/roy208/compressed-cryptonets/checkpoints/miniImagenet/ResNet18qat_dct/filter_8_pattern_default_input_64_56_56_bitwidth_4_relu1Prune_skipConnectPrune2/
resume=
num_classes=200
epochs=60
schedule_1=20
schedule_2=45
schedule_3=45
save_freq=10
batch_size=32
test_batch_size=64
num_workers=16
optimizer=adam
lr=0.001
weight_decay=1e-5
grad_clip_value=0.1
dropout=0.2
dct_status=Y
image_size=56
channels=64
filter_size=8
dct_pattern=default
bit_width=4
# ----------------------------
echo "dataset=${dataset}"
echo "num_classes=${num_classes}"
echo "model=${model}"
echo "epochs=${epochs}"
echo "schedule_1=${schedule_1}"
echo "schedule_2=${schedule_2}"
echo "save_freq=${save_freq}"
echo "image_size=${image_size}"
echo "batch_size=${batch_size}"
echo "test_batch_size=${test_batch_size}"
echo "num_workers=${num_workers}"
echo "lr=${lr}"
echo "weight_decay=${weight_decay}"
echo "grad_clip_value=${grad_clip_value}"
echo "dropout=${dropout}"
echo "dataset_path=${dataset_path}"
echo "checkpoint_dir=${checkpoint_dir}"
echo "resume=${resume}"
echo "bit_width=${bit_width}"
echo "dct_status=${dct_status}"


if [ "${dct_status}" == Y ]; then
  echo "filter_size=${filter_size}"
  echo "dct_pattern=${dct_pattern}"
  echo -e "channels=${channels}\n"
  CUDA_VISIBLE_DEVICES="${gpu}" python -u train_simple.py \
    --dataset "${dataset}" \
    --dataset_path "${dataset_path}" \
    --checkpoint_dir "${checkpoint_dir}" \
    --resume "${resume}" \
    --num_classes "${num_classes}" \
    --model "${model}" \
    --stop_epoch "${epochs}" \
    --schedule "${schedule_1}" "${schedule_2}" "${schedule_3}" \
    --save_freq "${save_freq}" \
    --image_size_dct "${image_size}" \
    --channels "${channels}" \
    --filter_size "${filter_size}" \
    --batch_size "${batch_size}" \
    --test_batch_size "${test_batch_size}" \
    --num_workers "${num_workers}" \
    --optimizer "${optimizer}" \
    --lr "${lr}" \
    --weight_decay "${weight_decay}" \
    --grad_clip_value "${grad_clip_value}" \
    --dropout "${dropout}" \
    --bit_width "${bit_width}" \
    --dct_pattern "${dct_pattern}" \
    --dct_status \
    --train_aug
else
  echo -e "\n"
  CUDA_VISIBLE_DEVICES="${gpu}" python -u train_simple.py \
    --dataset "${dataset}" \
    --dataset_path "${dataset_path}" \
    --checkpoint_dir "${checkpoint_dir}" \
    --resume "${resume}" \
    --num_classes "${num_classes}" \
    --model "${model}" \
    --stop_epoch "${epochs}" \
    --schedule "${schedule_1}" "${schedule_2}" "${schedule_3}" \
    --save_freq "${save_freq}" \
    --image_size "${image_size}" \
    --channels "${channels}" \
    --batch_size "${batch_size}" \
    --test_batch_size "${test_batch_size}" \
    --num_workers "${num_workers}" \
    --optimizer "${optimizer}" \
    --lr "${lr}" \
    --weight_decay "${weight_decay}" \
    --grad_clip_value "${grad_clip_value}" \
    --dropout "${dropout}" \
    --bit_width "${bit_width}" \
    --train_aug
fi
