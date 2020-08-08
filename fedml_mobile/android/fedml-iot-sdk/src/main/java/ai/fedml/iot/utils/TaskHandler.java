package ai.fedml.iot.utils;

import android.os.Handler;

public class TaskHandler extends Handler {
    public TaskHandler() {
        super(TaskManager.getInstance().taskLooper());
    }
}
