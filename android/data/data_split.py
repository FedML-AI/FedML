import idx2numpy
import os

N = 4

X_train = idx2numpy.convert_from_file('./client/mnist/train-images-idx3-ubyte')
y_train = idx2numpy.convert_from_file('./client/mnist/train-labels-idx1-ubyte')
X_test = idx2numpy.convert_from_file('./client/mnist/t10k-images-idx3-ubyte')
y_test = idx2numpy.convert_from_file('./client/mnist/t10k-labels-idx1-ubyte')

size = X_train.shape[0] // N

for i in range(N):
    os.makedirs(f'./client{i}/mnist', exist_ok=True)
    idx2numpy.convert_to_file(f'./client{i}/mnist/train-images-idx3-ubyte', X_train[i*size:(i+1)*size])
    idx2numpy.convert_to_file(f'./client{i}/mnist/train-labels-idx1-ubyte', y_train[i*size:(i+1)*size])
    idx2numpy.convert_to_file(f'./client{i}/mnist/t10k-images-idx3-ubyte', X_test)
    idx2numpy.convert_to_file(f'./client{i}/mnist/t10k-labels-idx1-ubyte', y_test)