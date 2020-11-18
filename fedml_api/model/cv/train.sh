CUDA_VISIBLE_DEVICES=0 python deeplab_train.py --backbone xception --lr 0.01 --workers 4 --epochs 20 --batch-size 6 --gpu-ids 0 --checkname deeplab-xception --eval-interval 2
