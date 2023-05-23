import idx2numpy
import os

N = 4

X_train = idx2numpy.convert_from_file('train-images-idx3-ubyte')
y_train = idx2numpy.convert_from_file('train-labels-idx1-ubyte')
X_test = idx2numpy.convert_from_file('t10k-images-idx3-ubyte')
y_test = idx2numpy.convert_from_file('t10k-labels-idx1-ubyte')

size = X_train.shape[0] // N

for i in range(N):
    os.makedirs(f'./client{i}', exist_ok=True)
    idx2numpy.convert_to_file(f'./client{i}/train-images-idx3-ubyte', X_train[i*size:(i+1)*size])
    idx2numpy.convert_to_file(f'./client{i}/train-labels-idx1-ubyte', y_train[i*size:(i+1)*size])
    idx2numpy.convert_to_file(f'./client{i}/t10k-images-idx3-ubyte', X_test)
    idx2numpy.convert_to_file(f'./client{i}/t10k-labels-idx1-ubyte', y_test)