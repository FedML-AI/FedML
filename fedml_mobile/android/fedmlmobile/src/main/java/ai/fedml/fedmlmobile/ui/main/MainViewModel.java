package ai.fedml.fedmlmobile.ui.main;

import android.content.Context;
import android.text.TextUtils;
import android.util.Log;

import androidx.annotation.NonNull;
import androidx.lifecycle.ViewModel;

import java.nio.charset.StandardCharsets;

import ai.fedml.fedmlsdk.FedMlMobileManager;
import ai.fedml.fedmlsdk.utils.StorageUtils;

public class MainViewModel extends ViewModel {
    public static final String TAG = "MainViewModel";

    public static final String TRAINING_EXECUTOR_BASE_URL = "http://192.168.3.104:5000";
    public static final String TRAINING_EXECUTOR_MQTT = "tcp://81.71.1.31:1883";

    void initial(@NonNull Context context) {
        FedMlMobileManager.getFedMlMobileApi().init(context, TRAINING_EXECUTOR_BASE_URL, TRAINING_EXECUTOR_MQTT);
        FedMlMobileManager.getFedMlMobileApi().setFedMlTaskListener((param) -> {
            Log.d(TAG, "FedMlTask:" + param);
        });
    }

    void upload() {
        String data = "FebML is very good!";
        final String modeFilePath = StorageUtils.saveToModelPath(data.getBytes(StandardCharsets.UTF_8), "model.txt");
        Log.d(TAG, "generateModelFile: " + modeFilePath);
        if (!TextUtils.isEmpty(modeFilePath)) {
            FedMlMobileManager.getFedMlMobileApi().uploadFile("model.txt", modeFilePath);
        }
    }

    void download() {
        FedMlMobileManager.getFedMlMobileApi().downloadFile("resource.html", TRAINING_EXECUTOR_BASE_URL
                + "/download/resource.html");
    }

    void sendMessage() {
        FedMlMobileManager.getFedMlMobileApi().sendMessage("Say hello @" + System.currentTimeMillis());
    }
}
