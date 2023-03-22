# Introduction

Shakespeare dataset is built from the collective works of William Shakespeare. This dataset is used to perform tasks of next character prediction. We loaded data from TensorFlow Federated (TFF) [shakespeare load_data API](https://www.tensorflow.org/federated/api_docs/python/tff/simulation/datasets/shakespeare/load_data) to get the unzipped raw data in h5 format saved it on [google drive](https://drive.google.com/drive/u/0/folders/10NvSW2AVQiHsTZbPzxbldd9b1QrOarcg). 

# Prepare Shakespeare Dataset

You can run the following script to download the dataset:

```
sh download_stackoverflow.sh
```

# Statistics of Shakespeare Dataset

Data partition is the same as [TFF](https://www.tensorflow.org/federated/api_docs/python/tff/simulation/datasets/shakespeare), with the following statistics.  Each client corresponds to a speaking role with at least two lines.

| DATASET     | TRAIN CLIENTS | TRAIN EXAMPLES | TEST CLIENTS | TEST EXAMPLES |
| ----------- | ------------- | -------------- | ------------ | ------------- |
| SHAKESPEARE | 715           | 16,068         | 715          | 2356          |

