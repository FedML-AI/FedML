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

## Dataloader
data_loader.py - for distributed computing and standalone simulation

mnist_mobile_preprocessor.py - for IoT/Mobile training. It splits the dataset into small files, so each client only needs to store a small file, which saves the memory cost on edge devices and also largely recudes the loading time.


## Mobile MNIST Data Preprocessing



path: feel_api/data_preprocessing/MNIST/mnist_mobile_preprocessor.py



### Command Line Execution

`python mnist_mobile_preprocessor.py --client_num_per_round 10 --comm_round 10`



### Output File & Structure

Output Directory: MNIST_mobile_zip

If client_num_per_round (worker number) = 2, comm_round = 8:

MNIST_mobile _zip

​							|- 0.zip (for device 0)

​									|-test/test.json  (8 data samples)

​									|-train/train.json (8 data samples)

​							|-1.zip (for device 1)

​									|-test/test.json  (8 data samples)

​									|-train/train.json (8 data samples)

### Client Sampling Example

For each round of sampling (2 workers, 8 rounds):

client_indexes = [993 859], [507 818], [37 726], [642 577], [544 515],[978 22],[778 334]

Then for device 0, the data includes:

["f_00993", "f_00507", "f_00037", "f_00642", "f_00698", "f_00544", "f_00978", "f_00778"]

For device 1, the data includes:

["f_00859", "f_00818", "f_00726", "f_00762", "f_00577", "f_00515", "f_00022", "f_00334"]
