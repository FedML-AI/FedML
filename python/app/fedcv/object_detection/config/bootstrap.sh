#!/bin/bash
pip install fedml
pip install matplotlib numpy opencv-python Pillow PyYAML scipy torch torchvision tqdm tensorboard seaborn pandas onnx onnxruntime onnx-simplifier thop pycocotools
PROJECT_DIR=$(cd $(dirname $0); cd ..; pwd)
cd $PROJECT_DIR
bash ./data/coco128/download_coco128_mlops.sh
