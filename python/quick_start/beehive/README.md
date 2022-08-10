# Install FedML
```
pip install fedml
```

Note that the mobile platform only supports Python 3.7 due to the contraints of MNN library.

# Start the server for FL in the mobile setting

1. adb push the data to your Android device
```
# for MAC OS
brew install android-platform-tools
../prepare.sh
```

2. Launch FedML Android App or SDK [https://github.com/FedML-AI/FedML/tree/master/android](https://github.com/FedML-AI/FedML/tree/master/android), and bind the Android Device to open.fedml.ai.

3. Check the device ID at open.fedml.ai (Edge Device)

4. Build Python Server Package and Upload to FedML MLOps Platform ("Create Application")

[https://github.com/FedML-AI/FedML/tree/master/python/quick_start/beehive](https://github.com/FedML-AI/FedML/tree/master/python/quick_start/beehive)

5. For local debugging of cross-device server, please try 

```
sh run_server.sh
```

6. Launch the training by using FedML MLOps (https://open.fedml.ai)

create group -> create project -> create run -> select application (the one we created for Android)
