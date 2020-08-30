package ai.febml.febmlmobile.ui.main;

import android.text.TextUtils;
import android.util.Log;

import androidx.lifecycle.ViewModel;

import com.google.common.io.Files;

import java.io.File;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Path;
import java.util.List;

import ai.febml.febmlmobile.trainingexecutor.TrainingExecutor;
import ai.febml.febmlmobile.utils.StorageUtils;

public class MainViewModel extends ViewModel {
    public static final String TAG = "MainViewModel";

    public static final String TRAINING_EXECUTOR_BASE_URL = "http://192.168.3.104:5000";
    public static final String TRAINING_EXECUTOR_MQTT = "tcp://81.71.1.31:1883";
    TrainingExecutor trainingExecutor = new TrainingExecutor(TRAINING_EXECUTOR_BASE_URL, TRAINING_EXECUTOR_MQTT);

    void initial() {
        trainingExecutor.init();
    }

    void upload() {
        String data = "FebML is very good!";
        final String modeFilePath = StorageUtils.saveToModelPath(data.getBytes(StandardCharsets.UTF_8), "model.txt");
        Log.d(TAG, "generateModelFile: " + modeFilePath);
        if (!TextUtils.isEmpty(modeFilePath)) {
            trainingExecutor.uploadFile("model.txt", modeFilePath);
        }
    }

    void download() {
        trainingExecutor.downloadFile("resource.html", TRAINING_EXECUTOR_BASE_URL
                + "/download/resource.html");
    }

    void sendMessage() {
        trainingExecutor.sendMessage("Say hello @" + System.currentTimeMillis());
    }
}
