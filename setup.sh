#!/bin/bash
set -e  # exit immediately if a command fails

# 1. Create datasets directory
mkdir -p all_datasets

# 2. Download dataset (Kaggle)
ZIP_PATH="$HOME/Downloads/ff-c23.zip"

curl -L -o "$ZIP_PATH" \
  https://www.kaggle.com/api/v1/datasets/download/xdxd003/ff-c23

# 3. Install unzip (system-level, NOT pip)
if ! command -v unzip &> /dev/null; then
  echo "unzip not found, installing..."
  sudo apt-get update
  sudo apt-get install -y unzip
fi

# 4. Unzip into all_datasets
unzip -q "$ZIP_PATH" -d all_datasets

echo "✅ Dataset downloaded and extracted to all_datasets/"
