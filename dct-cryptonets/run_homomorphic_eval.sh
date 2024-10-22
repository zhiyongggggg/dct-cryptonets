#!/usr/bin/env bash

# To run in the background:
# nohup bash run_homomorphic_eval.sh > {output file location} &

set -e
eval "$(conda shell.bash hook)"
conda activate dct-cryptonets
export BREVITAS_IGNORE_MISSING_KEYS=1

# ------ User Arguments ------
gpu=0
model=ResNet18qat
dataset=ImageNet
dataset_path=/home/datasets/imagenet/imagenet2012/
checkpoint_path=/home/arjunroy/Repos/dct-cryptonets/checkpoints/ImageNet/ResNet18qat_dct/filter_8_pattern_default_input_64_56_56_bitwidth_4/best.tar
num_classes=1000
verbose=True

fhe_mode=simulate
calib_batch_size=100
test_batch_size=50  # Should be set to 1 if fhe_mode=execute
test_subset=200  # Should be set to 1 if fhe_mode=execute
rounding_threshold_bits=6
n_bits=5
p_error=0.01
reliability_test=True

dct_status=Y
image_size=56
channels=64
filter_size=8
dct_pattern=default
bit_width=4
# ----------------------------
echo "model=${model}"
echo "dataset=${dataset}"
echo "num_classes=${num_classes}"
echo "image_size=${image_size}"
echo "calib_batch_size=${calib_batch_size}"
echo "test_batch_size=${test_batch_size}"
echo "test_subset=${test_subset}"
echo "fhe_mode=${fhe_mode}"
echo "dataset_path=${dataset_path}"
echo "checkpoint_path=${checkpoint_path}"
echo "bit_width=${bit_width}"
echo "rounding_threshold_bits=${rounding_threshold_bits}"
echo "n_bits=${n_bits}"
echo "p_error=${p_error}"
echo "reliability_test=${reliability_test}"
echo "verbose=${verbose}"
echo "dct_status=${dct_status}"


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
    --verbose "#${verbose}"
fi

