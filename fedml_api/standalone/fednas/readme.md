wandb login ee0b5f53d949c84cee7decbe7a629e63fb2f8408

## Experiments

Homogeneous distribution (IID) experiment:
```
# Learning Rate = 0.01; running starting from 21:39pm, Mar 9, 2020 (Monday), GPU card 3
nohup sh run_fednas_single_process_pytorch.sh 3 homo 50 5 0.01 > ./log/console_log_3.txt 2>&1 &

# Learning Rate = 0.001; running on Murali Node4 GPU1
nohup sh run_fednas_single_process_pytorch.sh 1 homo 50 5 0.001 > ./log/console_log_5_1.txt 2>&1 &

# learning rate = 0.03, Chaoyang's server, GPU 0
nohup sh run_fednas_single_process_pytorch.sh 0 hetero 50 5 0.3 > ./log/console_log_0.txt 2>&1 &
```

Heterogeneous distribution (Non-IID) experiment:
```
# Learning Rate = 0.01, hetero; running starting from 22:35pm, Mar 9, 2020 (Monday), Cyrus-Node2 GPU card 1
nohup sh run_fednas_single_process_pytorch.sh 1 hetero 50 5 0.01 > ./log/console_log_3.txt 2>&1 &

# Learning Rate = 0.001, heter; running on Murali's Node5 GPU1 
nohup sh run_fednas_single_process_pytorch.sh 1 hetero 50 5 0.001 > ./log/console_log_5_1.txt 2>&1 &

# learning rate = 0.03, Chaoyang's server, GPU 2
nohup sh run_fednas_single_process_pytorch.sh 2 hetero 50 5 0.3 > ./log/console_log_2.txt 2>&1 &
```
Test
```
sh run_fednas_single_process_pytorch.sh 0 hetero 50 5 0.3
```



Heterogeneous Data Distribution (worker number 16, randomly split the CIFAR10 dataset using the seed 0):
Data statistics: {0: {0: 541, 1: 213, 2: 34, 3: 179, 4: 13, 5: 99, 6: 20, 7: 37, 8: 107, 9: 189}, 1: {0: 9, 1: 3, 2: 92, 3: 20, 4: 438, 5: 1582, 6: 29, 7: 1158}, 2: {0: 5, 1: 119, 2: 64, 3: 461, 4: 431, 5: 753, 6: 208, 7: 1, 8: 7, 9: 9}, 3: {0: 225, 1: 148, 2: 1435, 3: 5, 4: 27, 5: 1085, 6: 11, 7: 112, 8: 170}, 4: {0: 286, 1: 769, 2: 320, 3: 342, 4: 855, 5: 234, 6: 1, 7: 301, 8: 8, 9: 28}, 5: {0: 717, 1: 65, 2: 4, 3: 908, 4: 1301, 5: 35, 6: 34, 7: 20, 8: 786}, 6: {0: 5, 1: 16, 2: 287, 3: 94, 4: 614, 5: 32, 6: 806, 8: 286, 9: 1789}, 7: {0: 410, 1: 1046, 2: 661, 3: 1002, 4: 9}, 8: {0: 640, 1: 114, 2: 62, 3: 71, 4: 309, 5: 47, 6: 553, 7: 771, 8: 534, 9: 154}, 9: {0: 2, 1: 2, 2: 1062, 3: 61, 4: 128, 5: 289, 6: 1014, 7: 852}, 10: {0: 659, 1: 355, 2: 278, 3: 569, 4: 71, 5: 362, 6: 137, 7: 652, 8: 434}, 11: {0: 294, 1: 283, 2: 187, 3: 93, 5: 239, 6: 130, 7: 825, 8: 787, 9: 817}, 12: {0: 262, 1: 746, 2: 18, 3: 104, 4: 9, 6: 315, 7: 11, 8: 9, 9: 111}, 13: {0: 279, 1: 266, 2: 77, 3: 33, 4: 13, 5: 223, 6: 834, 7: 236, 8: 885, 9: 345}, 14: {0: 510, 1: 19, 2: 21, 3: 951, 4: 27, 5: 8, 6: 493, 7: 22, 8: 963, 9: 1465}, 15: {0: 156, 1: 836, 2: 398, 3: 107, 4: 755, 5: 12, 6: 415, 7: 2, 8: 24, 9: 93}}
