package ai.fedml.fedmlsdk;

import android.content.Context;

import androidx.annotation.NonNull;

public interface FedMlMobileApi {
    /**
     * initialize FedMl Mobile Device
     *
     * @param context Context
     * @param baseUrl Executor
     * @param broker  broker server:port
     */
    void init(@NonNull final Context context, @NonNull final String baseUrl, @NonNull final String broker);

    /**
     * upload files,such as models.
     *
     * @param fileName the file name
     * @param filePath the file path
     * @return
     */
    boolean uploadFile(@NonNull final String fileName, @NonNull final String filePath);

    /**
     * download files,such as the data set.
     *
     * @param fileName the name to save
     * @param url      the utl of the file
     */
    void downloadFile(@NonNull final String fileName, @NonNull final String url);

    /**
     * send message to executor
     *
     * @param msg the message
     */
    void sendMessage(@NonNull String msg);

    /**
     * set Listener to get training task
     *
     * @param listener the listener
     */
    void setFedMlTaskListener(FedMlTaskListener listener);
}
