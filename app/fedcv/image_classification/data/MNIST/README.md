# Introduction
MNIST is a dataset to study image classification of handwritten digits 0-9 (LeCun et al., 1998). 

The original data source:
training set images: http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz
training set labels: http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz
test set images:     http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz
test set labels:     http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz

To simulate a heterogeneous setting, we distribute the data among 1000 devices such that each device
has samples of only 2 digits and the number of samples per device follows a power law. The input of the model is a
flattened 784-dimensional (28 × 28) image, and the output is a class label between 0 and 9.

This benchmark dataset is aligned with the following publication: Federated Optimization in Heterogeneous Networks (https://arxiv.org/pdf/1812.06127.pdf). MLSys 2020.

# Prepare MNIST Dataset

To simplify the data preparation, we provide processed non-I.I.D. dataset.
Please download the dataset [here](https://drive.google.com/file/d/1cU_LcBAUZvfZWveOMhG4G5Fg9uFXhVdf/view?usp=sharing), unzip it and put the `train` and `test` folder under this directory.
Or you can run the following script:
```
sh download_and_unzip.sh
```

The layout of the folders under `./MNIST` should be:

```
| MNIST

----| train 

---- ----| all_data_0_niid_0_keep_10_train_9.json

----| test

---- ----| all_data_0_niid_0_keep_10_train_9.json

| README.md
```
# Run Statistics

After download the dataset, please run the statistics to confirm that the distribution is the same as our preprocessing.

## Train
```
sh stats.sh train

####################################
DATASET: MNIST
1000 users
61664 samples (total)
61.66 samples per user (mean)
num_samples (std): 144.63
num_samples (std/mean): 2.35
num_samples (skewness): 8.18

num_sam num_users
0        447
20       241
40       105
60       50
80       33
100      20
120      15
140      9
160      9
180      11

```

## Test
```
sh stats.sh test

# result
####################################
DATASET: MNIST
1000 users
7371 samples (total)
7.37 samples per user (mean)
num_samples (std): 16.08
num_samples (std/mean): 2.18
num_samples (skewness): 8.20

num_sam num_users
0        927
20       40
40       19
60       8
80       1
100      0
120      1
140      2
160      0
180      1
```

