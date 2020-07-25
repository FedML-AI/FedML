original data source:
training set images: http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz
training set labels: http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz
test set images:     http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz
test set labels:     http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz

But in our paper, we use another source as follows:
https://drive.google.com/file/d/1cU_LcBAUZvfZWveOMhG4G5Fg9uFXhVdf/view?usp=sharing)
unzip it and put the json files into corresponding directory, train_file_name.json into "train" directory, and test_file_name.json into "test" directory

Distribution as follows:

DATASET: mnist
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

