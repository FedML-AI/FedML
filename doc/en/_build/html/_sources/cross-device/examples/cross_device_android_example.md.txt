# Federated Learning on Android Smartphones

Please follow this tutorial (https://doc.fedml.ai/mlops/user_guide.html) to start training using FedML BeeHive Platform.

<table>
<tr>
<td valign="center">
<img src="./../../_static/image/android_running.jpeg" alt="drawing" style="width:300px;"/>
</td>
<td valign="center">
<img src="./../../_static/image/edge_devices_overview.png" alt="drawing" style="width:400px;"/>
</td>
</tr>
</table>

This example will guide you to work through how to run federated learning on Android smartphones.
The code for this example locates in the following two paths:


Android Client (App) and SDK: [https://github.com/FedML-AI/FedML/tree/master/android](https://github.com/FedML-AI/FedML/tree/master/android)

Python Server: [https://github.com/FedML-AI/FedML/tree/master/python/quick_start/beehive](https://github.com/FedML-AI/FedML/tree/master/python/quick_start/beehive)

Android Client is an open source version. Check [this guidance](https://github.com/FedML-AI/FedML/blob/master/android/README.md) to understand the software architecture design. Next show you the step-by-step user experiment of using FedML Beehive.


## **1. Install Synthetic Data on Android Devices**

### 1.1 Install the adb command tool on your laptop

If you haven't installed `adb`, please refer to the installation steps at [https://www.xda-developers.com/install-adb-windows-macos-linux/](https://www.xda-developers.com/install-adb-windows-macos-linux/)

Then you should turn on the developer mode and USB debugging options for your Android device. The specific operation of each brand of device is not consistent; you can find and refer to the relevant instructions.

Next please connect the Android device to your laptop, and run the following command to see your device serial number.

```shell
adb devices
```
If it works correctly, it means you have successfully connected your laptop to your mobile device using adb.

### 1.2 Transferring data to mobile devices

You can download the required data and transfer it to the specified directory of the device with the following command:

```shell
bash prepare.sh
```

`prepare.sh` is as follows:

```shell
MNIST_DIR=mnist
CIFAR10_DIR=cifar10
ANDROID_DIR=/sdcard/ai.fedml

rm -rf $MNIST_DIR
mkdir $MNIST_DIR
wget http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz -P $MNIST_DIR
wget http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz -P $MNIST_DIR

rm -rf $CIFAR10_DIR
rm -rf cifar-10-binary.tar.gz
wget wget https://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz
tar -xzvf cifar-10-binary.tar.gz
mv cifar-10-batches-bin $CIFAR10_DIR

adb push $MNIST_DIR $ANDROID_DIR
adb push $CIFAR10_DIR $ANDROID_DIR
```

The MNIST dataset folder and the CIFAR-10 dataset folder can be moved to `/sdcard/ai` by running the above script. 

## 2. Get Started with FedML Android App

This section guides you through 1) installing Android Apk, 2) binding your Android smartphone devices to FedML MLOps Platform, and 3) set the data path for training.

After installing FedML Android App (https://github.com/FedML-AI/FedML/tree/master/android/app), please go to the MLOps platform (https://open.fedml.ai) - Beehive and switch to the `Edge Devices` page, you can see a list of **My Edge Devices** at the bottom, as well as a QR code and **Account ID:XXX** at the top right.

<img src="./../../_static/image/beehive-device.png" alt="image-20220427204703095" style="zoom:67%;" />

You can also see the binding devices in the **My Edge Devices** list on the web page.

To set data path on your device, click the top green bar. Set it as the path to the corresponding dataset moved to the Android device (find the folder name starting from ai.fedml).

#### 3. **Deploy FL Server**


- Build Python Server Package and Upload to FedML MLOps Platform ("Create Application")

[https://github.com/FedML-AI/FedML/tree/master/python/quick_start/beehive](https://github.com/FedML-AI/FedML/tree/master/python/quick_start/beehive)

- For local debugging of cross-device server, please try 

```
sh run_server.sh
```

- Launch the training by using FedML MLOps (https://open.fedml.ai)

Steps at MLOps: create group -> create project -> create run -> select application (the one we created for Android) -> start run

On the Android side, you will see training status as follows if every step works correctly.

<img src="./../../_static/image/android_running.jpeg" alt="image-20220428113930309" style="width:350px;margin:0 auto" />


## 3. Get Started with Integrating Android SDK for Your Host App

`android/fedmlsdk_demo` is a short tutorial for integrating Android SDK for your host App.

1. add repositories by maven

```groovy
    maven { url 'https://s01.oss.sonatype.org/content/repositories/snapshots' }
```

2. add dependency in build.gradle 

check `android/fedmlsdk_demo/build.gradle` as an example:

```groovy
    implementation 'ai.fedml:fedml-edge-android:1.0.0-SNAPSHOT'
```

3. add FedML account id to meta-data in AndroidManifest.xml

check `android/fedmlsdk_demo/src/main/AndroidManifest.xml` as an example:


```xml

<meta-data android:name="fedml_account" android:value="208" />
```

or

```xml

<meta-data android:name="fedml_account" android:resource="@string/fed_ml_account" />
```

You can find your account ID at FedML Open Platform (https://open.fedml.ai):
![account](./../../_static/image/beehive_account.png)

4. initial FedML Android SDK on your `Application` class.

Taking `android/fedmlsdk_demo/src/main/java/ai/fedml/edgedemo/App.java` as an example:
```java
package ai.fedml.edgedemo;

import android.app.Application;
import android.os.Handler;
import android.os.Looper;

import ai.fedml.edge.FedEdgeManager;

public class App extends Application {
    private static Handler sHandler = new Handler(Looper.getMainLooper());

    @Override
    public void onCreate() {
        super.onCreate();
        
        // initial Edge SDK
        FedEdgeManager.getFedEdgeApi().init(this);
        
        // set data path (to prepare data, please check this script `android/data/prepare.sh`)
        FedEdgeManager.getFedEdgeApi().setPrivatePath(Environment.getExternalStorageDirectory().getPath()
                + "/ai.fedml/device_1/user_0");
    }
}
```

## Android SDK APIs 
At the current stage, we provide high-level APIs with the following three classes.


- ai.fedml.edge.FedEdgeManager

This is the top APIs in FedML Android SDK, it supports core training engine and related control commands on your Android devices.

- ai.fedml.edge.OnTrainProgressListener

This is the message flow to interact between FedML Android SDK and your host APP.

- ai.fedml.edge.request.RequestManager

This is used to to connect your Android SDK with FedML Open Platform (https://open.fedml.ai), which helps you to simplify the deployment, edge collaborative training, experimental tracking, and more.

You can import them in your Java/Android projects as follows. See [android/fedmlsdk_demo/src/main/java/ai/fedml/edgedemo/ui/main/MainFragment.java](fedmlsdk_demo/src/main/java/ai/fedml/edgedemo/ui/main/MainFragment.java) as an example.
```
import ai.fedml.edge.FedEdgeManager;
import ai.fedml.edge.OnTrainProgressListener;
import ai.fedml.edge.request.RequestManager;
```


