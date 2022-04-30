# FedML-Mobile: Federated Learning Research Library for Android and iOS Smartphones (supported by FedML framework)

# Installation
http://doc.fedml.ai/#/installation

After the clone of this repository, please run the following command to get `FedML` submodule to your local.
```
cd FedML
git submodule init
git submodule update
```


# Update FedML Submodule
```
cd FedML
git checkout master && git pull
cd ..
git add FedML
git commit -m "updating submodule FedML to latest"
git push
```


# On-device training for Android

This project aims at training NN with non-IID dataset on Android device, involving doing training independently on device and updating NN in server using FedAvg Algorithm.  

We published a demo of training a LR model built with Deeplearning4J on Android Studio.  

Please check this out: https://github.com/FedML-AI/FedML-Mobile/tree/master/android/fedml-iot-sdk/src/main/java/ai/fedml/iot/service/FedML_client_training_demo
