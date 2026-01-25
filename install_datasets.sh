#!/usr/bin/env bash
set -e  # exit immediately on error

helpFunction() {
  echo ""
  echo "Usage: $0 -a Y/N -b Y/N -c Y/N -d DATASET_DIR"
  echo -e "\t-a Download ImageNette     (Y/N)"
  echo -e "\t-b Download mini-ImageNet  (Y/N)"
  echo -e "\t-c Download ImageNet-1K    (Y/N)"
  echo -e "\t-d Dataset root directory"
  exit 1
}

while getopts "a:b:c:d:" opt; do
  case "$opt" in
    a) parameterA="$OPTARG" ;;
    b) parameterB="$OPTARG" ;;
    c) parameterC="$OPTARG" ;;
    d) parameterD="$OPTARG" ;;
    ?) helpFunction ;;
  esac
done

if [ -z "$parameterA" ] || [ -z "$parameterB" ] || [ -z "$parameterC" ] || [ -z "$parameterD" ]; then
  echo "Missing required arguments"
  helpFunction
fi

ROOT_DIR=$(pwd)
DATASET_DIR="$parameterD"

# Ensure dataset root exists
mkdir -p "$DATASET_DIR"
cd "$DATASET_DIR" || { echo "Cannot access $DATASET_DIR"; exit 1; }

echo "Using dataset directory: $DATASET_DIR"

# --------------------------------------------------
# ImageNette
# --------------------------------------------------
if [ "$parameterA" == "Y" ]; then
  echo "ImageNette"

  mkdir -p ImageNette
  cd ImageNette

  if [ ! -f imagenette2.tgz ]; then
    wget https://s3.amazonaws.com/fast-ai-imageclas/imagenette2.tgz
  else
    echo "imagenette2.tgz already exists, skipping download"
  fi

  if [ ! -d imagenette2 ]; then
    tar -zxvf imagenette2.tgz
  else
    echo "ImageNette already extracted"
  fi

  cd "$DATASET_DIR"
fi

# --------------------------------------------------
# mini-ImageNet
# --------------------------------------------------
if [ "$parameterB" == "Y" ]; then
  echo "mini-ImageNet"

  mkdir -p miniImageNet
  cd miniImageNet

  for file in train.csv val.csv test.csv; do
    if [ ! -f "$file" ]; then
      wget https://github.com/twitter/meta-learning-lstm/raw/master/data/miniImagenet/$file
    else
      echo "$file already exists"
    fi
  done

  if [ ! -f ILSVRC2015_CLS-LOC.tar.gz ]; then
    wget http://image-net.org/image/ILSVRC2015/ILSVRC2015_CLS-LOC.tar.gz
  fi

  if [ ! -d ILSVRC2015 ]; then
    tar -zxvf ILSVRC2015_CLS-LOC.tar.gz
  fi

  cd "$ROOT_DIR"
  python compressed_cryptonetes/data/make_miniImagenet_json.py --dataset_dir "$DATASET_DIR"
  cd "$DATASET_DIR"
fi

# --------------------------------------------------
# ImageNet-1K
# --------------------------------------------------
if [ "$parameterC" == "Y" ]; then
  echo "ImageNet-1K"

  mkdir -p ImageNet
  cd ImageNet

  mkdir -p train val

  if [ ! -f train/ILSVRC2012_img_train.tar ]; then
    wget -O train/ILSVRC2012_img_train.tar \
      https://image-net.org/data/ILSVRC/2012/ILSVRC2012_img_train.tar
  else
    echo "Train tar already exists"
  fi

  if [ ! -f val/ILSVRC2012_img_val.tar ]; then
    wget -O val/ILSVRC2012_img_val.tar \
      https://image-net.org/data/ILSVRC/2012/ILSVRC2012_img_val.tar
  else
    echo "Val tar already exists"
  fi

  cd train
  if [ -z "$(ls -A . | grep -v tar)" ]; then
    tar -xvf ILSVRC2012_img_train.tar
    find . -name "*.tar" -exec sh -c 'mkdir -p "${0%.tar}" && tar -xf "$0" -C "${0%.tar}" && rm "$0"' {} \;
  else
    echo "Train already extracted"
  fi

  cd ../val
  if [ -z "$(ls -A . | grep -v tar)" ]; then
    tar -xvf ILSVRC2012_img_val.tar
    wget -qO- https://raw.githubusercontent.com/soumith/imagenetloader.torch/master/valprep.sh | bash
  else
    echo "Val already prepared"
  fi
fi

echo "All requested datasets processed safely."
