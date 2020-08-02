package ai.fedml.iot.utils;

import android.os.Handler;

public class BackgroundHandler extends Handler {
    public BackgroundHandler() {
        super(TaskManager.getInstance().backgroundLooper());
    }
}
