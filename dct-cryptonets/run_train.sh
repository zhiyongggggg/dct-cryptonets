#!/usr/bin/env bash

# To run in the background:
# nohup bash run_train_simple.sh > {output file location} &

set -e
eval "$(conda shell.bash hook)"
conda activate dct-cryptonets
export BREVITAS_IGNORE_MISSING_KEYS=1

# ------ User Arguments ------
# General parameters
gpu=0                  # Multi-GPU training is not currently supported with QAT Brevitas
model=ResNet18qat
dataset=ImageNet
num_classes=1000
dataset_path=/home/datasets/imagenet/imagenet2012/
checkpoint_dir=/home/arjunroy/Desktop/
resume=

# Training parameters
epochs=10
batch_size=256
test_batch_size=512
num_workers=4
optimizer=adam
lr=0.001
weight_decay=1e-5
grad_clip_value=0.1
dropout=0.2
schedule_1=5
schedule_2=10
schedule_3=10
checkpoint_save_freq=5
bit_width=4             # QAT trained bit-width. Set to 4 if cifar10, mini-ImageNet, Imagenette; otherwise 5 if ImageNet

# DCT parameters
dct_status=Y            # Set to N if running RGB-based network
image_size=56           # Set to 224 if running RGB-based network
channels=64             # Set to 3 if running RGB-based network
filter_size=8           # Set to 4 if running ResNet20 model; otherwise 8 if ResNet18 model
dct_pattern=default


echo "-----General parameters-----"
echo "model=${model}"
echo "dataset=${dataset}"
echo "num_classes=${num_classes}"
echo "dataset_path=${dataset_path}"
echo "checkpoint_dir=${checkpoint_dir}"
echo "resume=${resume}"

echo "-----Training parameters-----"
echo "epochs=${epochs}"
echo "batch_size=${batch_size}"
echo "test_batch_size=${test_batch_size}"
echo "num_workers=${num_workers}"
echo "optimizer=${optimizer}"
echo "lr=${lr}"
echo "weight_decay=${weight_decay}"
echo "grad_clip_value=${grad_clip_value}"
echo "dropout=${dropout}"
echo "schedule=[${schedule_1}, ${schedule_2}, ${schedule_3}]"
echo "checkpoint_save_freq=${checkpoint_save_freq}"
echo "bit_width=${bit_width}"

echo "-----DCT parameters-----"
echo "dct_status=${dct_status}"
echo "image_size=${image_size}"


if [ "${dct_status}" == Y ]; then
  echo "filter_size=${filter_size}"
  echo "dct_pattern=${dct_pattern}"
  echo -e "channels=${channels}\n"
  CUDA_VISIBLE_DEVICES="${gpu}" python -u train.py \
    --dataset "${dataset}" \
    --dataset_path "${dataset_path}" \
    --checkpoint_dir "${checkpoint_dir}" \
    --resume "${resume}" \
    --num_classes "${num_classes}" \
    --model "${model}" \
    --stop_epoch "${epochs}" \
    --schedule "${schedule_1}" "${schedule_2}" "${schedule_3}" \
    --save_freq "${checkpoint_save_freq}" \
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
  CUDA_VISIBLE_DEVICES="${gpu}" python -u train.py \
    --dataset "${dataset}" \
    --dataset_path "${dataset_path}" \
    --checkpoint_dir "${checkpoint_dir}" \
    --resume "${resume}" \
    --num_classes "${num_classes}" \
    --model "${model}" \
    --stop_epoch "${epochs}" \
    --schedule "${schedule_1}" "${schedule_2}" "${schedule_3}" \
    --save_freq "${checkpoint_save_freq}" \
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
