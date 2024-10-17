#!/usr/bin/env bash

helpFunction()
{
  echo ""
  echo "Usage: $0 -a parameterA -b parameterB -c parameterC -d parameterD"
  echo -e "\t-a Download and install ImageNette?     type:(Y/N)"
  echo -e "\t-b Download and install mini-ImageNet?  type:(Y/N)"
  echo -e "\t-c Download and install ImageNet?       type:(Y/N)"
  echo -e "\t-d Directory path for datasets          type:PATH"
  exit 1 # Exit script after printing help
}

while getopts "a:b:c:d:" opt
do
  case "$opt" in
    a ) parameterA="$OPTARG" ;;
    b ) parameterB="$OPTARG" ;;
    c ) parameterC="$OPTARG" ;;
    d ) parameterD="$OPTARG" ;;
    ? ) helpFunction ;; # Print helpFunction in case parameter is non-existent
  esac
done

# Print helpFunction in case parameters are empty
if [ -z "$parameterA" ] || [ -z "$parameterB" ] || [ -z "$parameterC" ] || [ -z "$parameterD" ]
then
  echo "Some or all of the parameters are empty";
  helpFunction
fi

# Set dataset directory
ROOT_DIR=${pwd}
DATASET_DIR=${parameterD}
cd DATASET_DIR

# ImageNette
if [ "${parameterA}" == Y ]; then
  echo -e "\n"
  echo "Downloading and installing ImageNette..."
  mkdir ImageNette
  cd ImageNette
  wget https://s3.amazonaws.com/fast-ai-imageclas/imagenette2.tgz
  tar -zxvf imagenette2.tgz
  cd DATASET_DIR
fi

# miniImageNet
if [ "${parameterB}" == Y ]; then
  echo -e "\n"
  echo "Downloading and installing miniImageNet..."
  mkdir miniIamgeNet
  cd miniImagenet
  wget https://github.com/twitter/meta-learning-lstm/blob/master/data/miniImagenet/test.csv
  wget https://github.com/twitter/meta-learning-lstm/blob/master/data/miniImagenet/train.csv
  wget https://github.com/twitter/meta-learning-lstm/blob/master/data/miniImagenet/val.csv
  wget http://image-net.org/image/ILSVRC2015/ILSVRC2015_CLS-LOC.tar.gz
  tar -zxvf ILSVRC2015_CLS-LOC.tar.gz

  cd ROOT_DIR
  python compressed_cryptonetes/data/make_miniImagenet_json.py --dataset_dir DATASET_DIR
  cd DATASET_DIR
fi

# ImageNet-1K
#  train/
#  ├── n01440764
#  │   ├── n01440764_10026.JPEG
#  │   ├── n01440764_10027.JPEG
#  │   ├── ......
#  ├── ......
#  val/
#  ├── n01440764
#  │   ├── ILSVRC2012_val_00000293.JPEG
#  │   ├── ILSVRC2012_val_00002138.JPEG
#  │   ├── ......
#  ├── ......
if [ "${parameterC}" == Y ]; then
  echo -e "\n"
  echo "Downloading and installing ImageNet-1K (ILSVRC2012)..."
  # Download the data
  # get ILSVRC2012_img_val.tar (about 6.3 GB). MD5: 29b22e2961454d5413ddabcf34fc5622
  wget https://image-net.org/data/ILSVRC/2012/ILSVRC2012_img_val.tar
  # get ILSVRC2012_img_train.tar (about 138 GB). MD5: 1d675b47d978889d74fa0da5fadfb00e
  wget https://image-net.org/data/ILSVRC/2012/ILSVRC2012_img_train.tar

  # Extract the training data
  mkdir ImageNet
  cd ImageNet
  mkdir train && mv ILSVRC2012_img_train.tar train/ && cd train
  tar -xvf ILSVRC2012_img_train.tar && rm -f ILSVRC2012_img_train.tar
  find . -name "*.tar" | while read NAME ; do mkdir -p "${NAME%.tar}"; tar -xvf "${NAME}" -C "${NAME%.tar}"; rm -f "${NAME}"; done
  cd ..

  # Extract the validation data and move images to subfolders:
  mkdir val && mv ILSVRC2012_img_val.tar val/ && cd val && tar -xvf ILSVRC2012_img_val.tar
  wget -qO- https://raw.githubusercontent.com/soumith/imagenetloader.torch/master/valprep.sh | bash
fi
