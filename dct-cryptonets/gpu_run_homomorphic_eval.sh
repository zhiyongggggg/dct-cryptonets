#!/usr/bin/env bash

# To run in the background:
# nohup bash run_homomorphic_eval.sh > {output file location} &

set -e
eval "$(conda shell.bash hook)"
conda activate zlim135env_gpu_dctcryptonets
export BREVITAS_IGNORE_MISSING_KEYS=1

# ====== GPU Configuration ======
export CUDA_VISIBLE_DEVICES=0              # Select GPU device
export CONCRETE_USE_GPU=1                  # Enable GPU for Concrete
export CONCRETE_GPU_MEMORY_FRACTION=0.8    # Use 80% of GPU memory

# Fix CUDA library paths for system CUDA installation
export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/usr/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

# ------ User Arguments ------
# General parameters
gpu=0
num_classes=10
dataset_path=/hdd/zlim135/Git/dct-cryptonets/all_dataset/ImageNette/imagenette2/
checkpoint_path=/hdd/zlim135/Git/dct-cryptonets/checkpoint_dir/ImageNette/imagenette2/best.tar
bit_width=4

# Homomorphic encryption parameters
fhe_mode=simulate          # Use 'execute' for actual FHE with GPU benefits
calib_batch_size=100
test_batch_size=1          # Keep at 1 for 'execute' mode
test_subset=10             # Start small to test GPU
rounding_threshold_bits=6
n_bits=5
p_error=0.01
reliability_test=False     # Set to False for initial GPU testing
verbose=True

# DCT parameters
dct_status=Y
dct_pattern=default

# ResNet50 ImageNet with DCT
dataset=ImageNet
model=ResNet50qat
image_size=224
channels=64
filter_size=8

echo "=========================================="
echo "GPU Configuration"
echo "=========================================="
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "CONCRETE_USE_GPU=${CONCRETE_USE_GPU}"
echo "CONCRETE_GPU_MEMORY_FRACTION=${CONCRETE_GPU_MEMORY_FRACTION}"
echo ""

echo "=========================================="
echo "General Parameters"
echo "=========================================="
echo "model=${model}"
echo "dataset=${dataset}"
echo "num_classes=${num_classes}"
echo "dataset_path=${dataset_path}"
echo "checkpoint_path=${checkpoint_path}"
echo "bit_width=${bit_width}"
echo ""

echo "=========================================="
echo "Homomorphic Encryption Parameters"
echo "=========================================="
echo "fhe_mode=${fhe_mode}"
echo "calib_batch_size=${calib_batch_size}"
echo "test_batch_size=${test_batch_size}"
echo "test_subset=${test_subset}"
echo "rounding_threshold_bits=${rounding_threshold_bits}"
echo "n_bits=${n_bits}"
echo "p_error=${p_error}"
echo "reliability_test=${reliability_test}"
echo "verbose=${verbose}"
echo ""

echo "=========================================="
echo "DCT Parameters"
echo "=========================================="
echo "dct_status=${dct_status}"
echo "image_size=${image_size}"
echo "filter_size=${filter_size}"
echo "dct_pattern=${dct_pattern}"
echo "channels=${channels}"
echo ""

# Check GPU availability
echo "=========================================="
echo "GPU Check"
echo "=========================================="
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv
echo ""

if [ "${dct_status}" == Y ]; then
  CUDA_VISIBLE_DEVICES="${gpu}" python -u gpu_homomorphic_eval.py \
    --dataset "${dataset}" \
    --checkpoint_dir "${checkpoint_dir}" \
    --model "${model}" \
    --num_classes "${num_classes}" \
    --dataset_path "${dataset_path}" \
    --image_size_dct "${image_size}" \
    --channels "${channels}" \
    --filter_size "${filter_size}" \
    --fhe_mode "${fhe_mode}" \
    --calib_batch_size "${calib_batch_size}" \
    --test_batch_size "${test_batch_size}" \
    --bit_width "${bit_width}" \
    --dct_pattern "${dct_pattern}" \
    --test_subset "${test_subset}" \
    --rounding_threshold_bits "${rounding_threshold_bits}" \
    --n_bits "${n_bits}" \
    --p_error "${p_error}" \
    --reliability_test "${reliability_test}" \
    --verbose "${verbose}" \
    --dct_status
else
  CUDA_VISIBLE_DEVICES="${gpu}" python -u homomorphic_eval.py \
    --dataset "${dataset}" \
    --checkpoint_dir "${checkpoint_dir}" \
    --model "${model}" \
    --num_classes "${num_classes}" \
    --checkpoint_path "${checkpoint_path}" \
    --dataset_path "${dataset_path}" \
    --image_size "${image_size}" \
    --channels "${channels}" \
    --fhe_mode "${fhe_mode}" \
    --calib_batch_size "${calib_batch_size}" \
    --test_batch_size "${test_batch_size}" \
    --bit_width "${bit_width}" \
    --test_subset "${test_subset}" \
    --rounding_threshold_bits "${rounding_threshold_bits}" \
    --n_bits "${n_bits}" \
    --p_error "${p_error}" \
    --reliability_test "${reliability_test}" \
    --verbose "${verbose}"
fi