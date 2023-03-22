#!/usr/bin/env bash
python3 main_fedml_image_classification.py --cf config/fedml_config.yaml --run_id mobilenetv3_cifar10 --rank 0 --role server
