package ai.fedml.edgedemo;

import android.app.Application;
import android.os.Environment;
import android.os.Handler;
import android.os.Looper;

import ai.fedml.edge.FedEdgeManager;
import ai.fedml.edge.utils.StorageUtils;

public class App extends Application {
    private static final Handler sHandler = new Handler(Looper.getMainLooper());

    @Override
    public void onCreate() {
        super.onCreate();

        // Init FedML Android SDK
        FedEdgeManager.getFedEdgeApi().init(this);

        // set data path (to prepare data, please check this script `android/data/prepare.sh`)
        // e.g., /storage/emulated/0/Android/data/ai.fedml.edgedemo/files/
        FedEdgeManager.getFedEdgeApi().setPrivatePath(StorageUtils.getDatasetPath());
    }

    public static void runOnUiThread(Runnable runnable) {
        sHandler.post(runnable);
    }
}
