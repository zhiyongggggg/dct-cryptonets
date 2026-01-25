#!/usr/bin/env bash

# To run in the background:
# nohup bash run_homomorphic_eval.sh > {output file location} &

set -e
eval "$(conda shell.bash hook)"
conda activate dct-cryptonets
export BREVITAS_IGNORE_MISSING_KEYS=1

# ------ User Arguments ------
# General parameters
gpu=0
model=ResNet18qat
dataset=FaceForensic
num_classes=2
dataset_path=/home/zhiyong/Git/dct-cryptonets/dct-cryptonets/dataset/FaceForensic
checkpoint_path=/home/zhiyong/Git/dct-cryptonets/dct-cryptonets/checkpoint/faceforensic/best.tar
bit_width=4                 # QAT trained bit-width. Set to 4 if cifar10, mini-ImageNet, Imagenette; otherwise 5 if ImageNet

# Homomorphic encryption parameters (Dataset batch/subset sizes are dependent on this)
# FaceForensic Execute Mode Params
#fhe_mode=execute
#calib_batch_size=5
#test_batch_size=1          # Set to 1 if fhe_mode=execute
#test_subset=1             # Set to 1 if fhe_mode=execute; ~200 if fhe_mode=simulate and running reliability_test

# Simulate Mode Param
fhe_mode=simulate
calib_batch_size=100
test_batch_size=50
test_subset=200
rounding_threshold_bits=6   # Set to 6 if running cifar10, mini-ImageNet, Imagenette; otherwise 7 if ImageNet
n_bits=5
p_error=0.01
reliability_test=True
verbose=True

# DCT parameters
dct_status=Y                # Set to N if running RGB-based model
image_size=56               # Set to 224 if running RGB-based model
channels=24                 # Set to 3 if running RGB-based model
filter_size=8               # Set to 4 if running ResNet20 model; otherwise 8 if ResNet18 model
dct_pattern=default


echo "-----General parameters-----"
echo "model=${model}"
echo "dataset=${dataset}"
echo "num_classes=${num_classes}"
echo "dataset_path=${dataset_path}"
echo "checkpoint_path=${checkpoint_path}"
echo "bit_width=${bit_width}"

echo "-----Homomorphic encryption parameters-----"
echo "fhe_mode=${fhe_mode}"
echo "calib_batch_size=${calib_batch_size}"
echo "test_batch_size=${test_batch_size}"
echo "test_subset=${test_subset}"
echo "rounding_threshold_bits=${rounding_threshold_bits}"
echo "n_bits=${n_bits}"
echo "p_error=${p_error}"
echo "reliability_test=${reliability_test}"
echo "verbose=${verbose}"

echo "-----DCT parameters-----"
echo "dct_status=${dct_status}"
echo "image_size=${image_size}"


if [ "${dct_status}" == Y ]; then
  echo "filter_size=${filter_size}"
  echo "dct_pattern=${dct_pattern}"
  echo -e "channels=${channels}\n"
  CUDA_VISIBLE_DEVICES="${gpu}" python -u homomorphic_eval.py \
    --dataset "${dataset}" \
    --model "${model}" \
    --num_classes "${num_classes}" \
    --checkpoint_path "${checkpoint_path}" \
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
  echo -e "\n"
  CUDA_VISIBLE_DEVICES="${gpu}" python -u homomorphic_eval.py \
    --dataset "${dataset}" \
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
