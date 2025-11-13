#!/usr/bin/env bash

# To run in the background:
# nohup bash run_train_simple.sh > {output file location} &

set -e
eval "$(conda shell.bash hook)"
conda activate zlim135env_gpu_dctcryptonets
export BREVITAS_IGNORE_MISSING_KEYS=1

# ====== Clear GPU Memory First ======
echo "Clearing GPU memory..."
nvidia-smi
python -c "import torch; torch.cuda.empty_cache(); print('GPU cache cleared')"

# Optional: Kill previous Python processes (use with caution!)
# pkill -9 python || true

# Set memory allocation strategy
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

# ------ User Arguments ------
# General parameters
gpu=0
model=ResNet18qat
dataset=FaceForensic
num_classes=2
dataset_path=/hdd/zlim135/Git/dct-cryptonets/all_dataset/FaceForensic/postprocessed/
checkpoint_dir=/hdd/zlim135/Git/dct-cryptonets/checkpoint_dir/FaceForensic/postprocessed_dct/
resume=

# Training parameters
epochs=10
batch_size=8           # REDUCED from 16 to 8 to save memory
test_batch_size=16     # REDUCED from 32 to 16
num_workers=2          # REDUCED from 4 to 2 to save memory
optimizer=adam
lr=0.001
weight_decay=1e-5
grad_clip_value=0.1
dropout=0.2
schedule_1=5
schedule_2=10
schedule_3=10
checkpoint_save_freq=5
bit_width=4

# DCT parameters
dct_status=Y
image_size=224
channels=6
filter_size=8
dct_pattern=default

echo "====================================="
echo "GPU Memory Status Before Training"
echo "====================================="
nvidia-smi --query-gpu=memory.used,memory.free,memory.total --format=csv

echo ""
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

echo ""
echo "====================================="
echo "GPU Memory Status After Training"
echo "====================================="
nvidia-smi --query-gpu=memory.used,memory.free,memory.total --format=csv