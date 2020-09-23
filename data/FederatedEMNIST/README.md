# Introduction

EMNIST dataset extends MNIST dataset with upper and lower case English characters. We loaded data from TensorFlow Federated (TFF) [emnist load_data API](https://www.tensorflow.org/federated/api_docs/python/tff/simulation/datasets/emnist/load_data)  and saved it on [google drive]( https://drive.google.com/drive/folders/1S377qFHM_q_o1hE7-ODlmtRT6iV6N7AT?usp=sharing) in h5 format. 

# Prepare FederatedEMNIST Dataset

You can run the following script to download the dataset:

```
sh download_federatedEMNIST.sh
```

By default, it will download EMNIST dataset which includes digits and characters.  TFF also supports getting only the digits data from EMNIST dataset. So we generated another digits only dataset from it. To obtain the digits only dataset, use the following command to download it:

```
sh download_federatedEMNIST.sh digit
```

Or with Tensorflow dependencies, you can run this to process the data from Tensorflow locally:

```
python dataset.py
```

