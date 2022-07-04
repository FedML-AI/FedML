# Data Preprocessing

## Preprocessor
Preprocessor is one of the important classes in FedNLP. Preprocessor is in charge of preprocessing and transforming the raw data into trainnable features. Distinctive datasets have distinctive preprocessors. For example, some datasets in the NLP domain are required to do the tokenization but others are not. Some datasets need to use regular expression or other techniques to clean the text but others are not. The preprocessor class is accountable for preparing the input for the model(transform texts to numbers).

## utils
For each task formulation, we have a specific utils file to support preprocessing and training. For example, the utils file can include fucntions related to metrics, features and so on.