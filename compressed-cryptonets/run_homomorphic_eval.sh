#!/usr/bin/env bash

# To run in the background:
# nohup bash run_homomorphic_eval.sh > {output file location} &

set -e
eval "$(conda shell.bash hook)"
conda activate compressed-cryptonets
export BREVITAS_IGNORE_MISSING_KEYS=1

# ------ User Arguments ------
gpu=2
model=ResNet18qat
dataset=ImageNet
dataset_path=/local/a/imagenet/imagenet2012/
#dataset_path=/home/min/a/roy208/datasets/miniImagenet/
#checkpoint_path=/home/nano01/a/roy208/compressed-cryptonets/checkpoints/ImageNet/ResNet18qat_dct/filter_8_pattern_default_input_64_56_56_bitwidth_4/best.tar
checkpoint_path=/home/nano01/a/roy208/compressed-cryptonets/checkpoints/ImageNet/ResNet18qat/input_3_448_448_bitwidth_4/best.tar
#checkpoint_path=/home/nano01/a/roy208/compressed-cryptonets/checkpoints/Imagenette/ResNet18qat_dct/filter_8_pattern_default_input_64_56_56_bitwidth_4/best.tar
#checkpoint_path=/home/nano01/a/roy208/compressed-cryptonets/checkpoints/miniImagenet/ResNet18qat/input_3_224_224_bitwidth_4/best.tar
num_classes=1000
batch_size=100
test_batch_size=25

fhe_mode=simulate
test_subset=200        # Should be set to 1 if fhe_mode=execute
memory_profiler=N
mprof_directory=/home/arjunroy/Repos/compressed-cryptonets/results/homomorphic_eval/miniImagenet/memory_profiler/

#dct_status=Y
#image_size=56
#channels=64
dct_status=N
image_size=448
channels=3
filter_size=8
dct_pattern=default
bit_width=4

rounding_threshold_bits=6
n_bits=5
p_error=0.01

#model=PolyKervNet
#dataset=cifar10
#dct_status=N
#image_size=224
#channels=3
#batch_size=64
#fhe_mode=simulate
#test_subset=50        # Should be set to 1 if fhe_mode=execute
#dataset_path=/home/arjunroy/datasets/cifar10
#checkpoint_path=/home/arjunroy/Repos/PolyKervNets/model.pth
#memory_profiler=N
#mprof_directory=/home/arjunroy/Repos/compressed-cryptonets/results/homomorphic_eval/miniImagenet/memory_profiler/
# ----------------------------
echo "model=${model}"
echo "dataset=${dataset}"
echo "num_classes=${num_classes}"
echo "image_size=${image_size}"
echo "batch_size=${batch_size}"
echo "test_batch_size=${test_batch_size}"
echo "test_subset=${test_subset}"
echo "fhe_mode=${fhe_mode}"
echo "dataset_path=${dataset_path}"
echo "checkpoint_path=${checkpoint_path}"
echo "bit_width=${bit_width}"
echo "rounding_threshold_bits=${rounding_threshold_bits}"
echo "n_bits=${n_bits}"
echo "p_error=${p_error}"
echo "dct_status=${dct_status}"

if [ "${memory_profiler}" == Y ]; then
  mprof_output="$mprof_directory"/"${model}"_DCT_"${dct_status}"_img"${image_size}".dat
  rm -f "${mprof_output}"  # Otherwise file is appended to
  if [ "${dct_status}" == Y ]; then
    echo "filter_size=${filter_size}"
    echo "dct_pattern=${dct_pattern}"
    echo -e "channels=${channels}\n"
    mprof run \
      --interval 10 \
      --output "${mprof_output}" \
      python homomorphic_eval.py \
        --dataset "${dataset}" \
        --model "${model}" \
        --num_classes "${num_classes}" \
        --checkpoint_path "${checkpoint_path}" \
        --dataset_path "${dataset_path}" \
        --image_size_dct "${image_size}" \
        --channels "${channels}" \
        --filter_size "${filter_size}" \
        --fhe_mode "${fhe_mode}" \
        --batch_size "${batch_size}" \
        --test_batch_size "${test_batch_size}" \
        --bit_width "${bit_width}" \
        --dct_pattern "${dct_pattern}" \
        --test_subset "${test_subset}" \
        --rounding_threshold_bits "${rounding_threshold_bits}" \
        --n_bits "${n_bits}" \
        --p_error "${p_error}" \
        --dct_status
  else
    mprof run \
      --interval 10 \
      --output "${mprof_output}" \
      python homomorphic_eval.py \
        --dataset "${dataset}" \
        --model "${model}" \
        --num_classes "${num_classes}" \
        --checkpoint_path "${checkpoint_path}" \
        --dataset_path "${dataset_path}" \
        --image_size "${image_size}" \
        --channels "${channels}" \
        --fhe_mode "${fhe_mode}" \
        --batch_size "${batch_size}" \
        --test_batch_size "${test_batch_size}" \
        --bit_width "${bit_width}" \
        --test_subset "${test_subset}" \
        --rounding_threshold_bits "${rounding_threshold_bits}" \
        --n_bits "${n_bits}" \
        --p_error "${p_error}"
  fi
  echo -e "\nPeak Memory Usage: "
  mprof peak "${mprof_output}"
else
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
      --batch_size "${batch_size}" \
      --test_batch_size "${test_batch_size}" \
      --bit_width "${bit_width}" \
      --dct_pattern "${dct_pattern}" \
      --test_subset "${test_subset}" \
      --rounding_threshold_bits "${rounding_threshold_bits}" \
      --n_bits "${n_bits}" \
      --p_error "${p_error}" \
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
      --batch_size "${batch_size}" \
      --test_batch_size "${test_batch_size}" \
      --bit_width "${bit_width}" \
      --test_subset "${test_subset}" \
      --rounding_threshold_bits "${rounding_threshold_bits}" \
      --n_bits "${n_bits}" \
      --p_error "${p_error}"
  fi
fi
