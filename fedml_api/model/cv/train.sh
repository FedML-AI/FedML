CUDA_VISIBLE_DEVICES=0,1,2,3 python deeplab_train.py --backbone xception --lr 0.01 --workers 4 --epochs 30 --batch-size 16 --gpu-ids 0,1,2,3 --checkname deeplab-exception --eval-interval 1 
