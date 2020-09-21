package ai.fedml.fedmlsdk;

import android.content.Context;

import androidx.annotation.NonNull;

import ai.fedml.fedmlsdk.trainingexecutor.TrainingExecutor;

class FedMlMobileImpl implements FedMlMobileApi {
    private TrainingExecutor mTrainingExecutor;

    @Override
    public void init(@NonNull Context context, @NonNull String baseUrl, @NonNull String broker) {
        ContextHolder.initialize(context);
        registerDevice(baseUrl, broker);
    }

    private void registerDevice(@NonNull final String baseUrl, @NonNull final String broker) {
        mTrainingExecutor = new TrainingExecutor(baseUrl, broker);
        mTrainingExecutor.init();
    }

    @Override
    public boolean uploadFile(@NonNull final String fileName, @NonNull final String filePath) {
        mTrainingExecutor.uploadFile(fileName, filePath);
        return true;
    }

    @Override
    public void downloadFile(@NonNull String fileName, @NonNull String url) {
        mTrainingExecutor.downloadFile(fileName, url);
    }

    @Override
    public void sendMessage(@NonNull String msg) {
        mTrainingExecutor.sendMessage(msg);
    }

    @Override
    public void setFedMlTaskListener(FedMlTaskListener listener) {
        mTrainingExecutor.setFedMlTaskListener(listener);
    }
}
