MNIST_DIR=mnist
CIFAR10_DIR=cifar10
ANDROID_DIR=/sdcard/ai.fedml

rm -rf $MNIST_DIR
mkdir $MNIST_DIR
wget http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz -P $MNIST_DIR
wget http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz -P $MNIST_DIR

rm -rf $CIFAR10_DIR
rm -rf cifar-10-binary.tar.gz
wget https://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz
tar -xzvf cifar-10-binary.tar.gz
mv cifar-10-batches-bin $CIFAR10_DIR

adb push $MNIST_DIR $ANDROID_DIR
adb push $CIFAR10_DIR $ANDROID_DIR
