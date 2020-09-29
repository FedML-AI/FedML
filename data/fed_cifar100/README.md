# Introduction

We loaded data from TensorFlow Federated (TFF) [cifar100 load_data API](https://www.tensorflow.org/federated/api_docs/python/tff/simulation/datasets/cifar100/load_data)  and saved it on [google drive](https://drive.google.com/drive/folders/121SiMZj9WJMRNZHTkA1bBfBs-gPWG5nQ) in h5 format. 

# Prepare Cifar100 Dataset

You can run the following script to download the dataset:

```
sh download_fedcifar100.sh
```

Or with Tensorflow dependencies, you can run this to process the data from Tensorflow locally:

```
python dataset.py
```

