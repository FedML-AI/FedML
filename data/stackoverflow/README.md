# Introduction

Stack Overflow dataset hosted by Kaggle consists of questions and answers from the website Stack Overflow. This dataset is used to perform two tasks: tag prediction via logistic regression and next word prediction. We loaded data from TensorFlow Federated (TFF) [stackoverflow load_data API](https://www.tensorflow.org/federated/api_docs/python/tff/simulation/datasets/stackoverflow/load_data) to get the unzipped raw data in h5 format saved it on [google drive](https://drive.google.com/drive/folders/1-zQivrESzi8GMPMql57mWf0qJ5FCp1cK). 

# Prepare Stack Overflow Dataset

You can run the following script to download the dataset:

```
sh download_stackoverflow.sh
```

Or with Tensorflow dependencies, you can run this to process the data from Tensorflow locally:

```
python dataset.py
```

# Statistics of Stack Overflow Dataset

Data partition is the same as [TFF](https://www.tensorflow.org/federated/api_docs/python/tff/simulation/datasets/stackoverflow), with following statistics:

| DATASET       | TRAIN CLIENTS | TRAIN EXAMPLES | TEST CLIENTS | TEST EXAMPLES |
| ------------- | ------------- | -------------- | ------------ | ------------- |
| STACKOVERFLOW | 342,477       | 135,818,730    | 204,088      | 16,586,035    |

