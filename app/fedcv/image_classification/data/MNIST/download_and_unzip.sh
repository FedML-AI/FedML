rm -rf mnist
rm -rf train
rm -rf test

wget --no-check-certificate --no-proxy https://fedcv.s3.us-west-1.amazonaws.com/MNIST.zip

unzip MNIST.zip

mv mnist/train train
mv mnist/test test
rm -rf mnist
rm -rf MNIST.zip