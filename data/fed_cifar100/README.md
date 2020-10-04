# Introduction

We loaded data from TensorFlow Federated (TFF) [cifar100 load_data API](https://www.tensorflow.org/federated/api_docs/python/tff/simulation/datasets/cifar100/load_data)  and saved it on [google drive](https://drive.google.com/drive/folders/121SiMZj9WJMRNZHTkA1bBfBs-gPWG5nQ) in h5 format. 

# Prepare CIFAR100 Dataset

You can run the following script to download the dataset:

```
sh download_fedcifar100.sh
```

Or with Tensorflow dependencies, you can run this to process the data from Tensorflow locally:

```
python dataset.py
```

# Statistics of CIFAR100 Dataset

Data partition is the same as [TFF](https://www.tensorflow.org/federated/api_docs/python/tff/simulation/datasets/cifar100), with the following statistics.  

| DATASET   | TRAIN CLIENTS | TRAIN EXAMPLES | TEST CLIENTS | TEST EXAMPLES |
| --------- | ------------- | -------------- | ------------ | ------------- |
| EMNIST-62 | 500           | 50,000         | 100          | 10,000        |

