# Introduction
We reuse the dataset preprocessing for Shakespeare published by FedProx (https://arxiv.org/abs/1812.06127).

# Shakespeare Dataset

You can download the training and testing datasets [here](https://drive.google.com/drive/folders/1jKREFyb4SGLnh8GE5jM6WX3AY_qiEKsY), put them  at `train` and `test` folder under this directory.

Or you can directly use the following command to download it manually:
```
sh download_shakespeare.sh.sh
```


# Run Statistics

After download the dataset, please run the statistics to confirm that the distribution is the same as our preprocessing.

## Train
```
sh stats.sh train

# Result
####################################
DATASET: shakespeare
143 users
413629 samples (total)
2892.51 samples per user (mean)
num_samples (std): 5446.75
num_samples (std/mean): 1.88
num_samples (skewness): 3.27

num_sam num_users
0        99
2000     15
4000     7
6000     8
8000     3
10000    3
12000    2
14000    0
16000    0
18000    2
```

## Test
```
sh stats.sh test

# result
####################################
DATASET: shakespeare
143 users
103477 samples (total)
723.62 samples per user (mean)
num_samples (std): 1361.69
num_samples (std/mean): 1.88
num_samples (skewness): 3.27

num_sam num_users
0        129
2000     8
4000     2
6000     3
8000     1
10000    0
12000    0
14000    0
16000    0
18000    0

```

