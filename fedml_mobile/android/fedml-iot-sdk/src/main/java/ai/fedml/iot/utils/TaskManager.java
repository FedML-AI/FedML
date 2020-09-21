package ai.fedml.iot.utils;

import android.os.HandlerThread;
import android.os.Looper;

public class TaskManager {

    private volatile static TaskManager sInstance;

    private volatile HandlerThread mTaskHandlerThread;
    private volatile HandlerThread mBackgroudHandlerThread;

    private TaskManager() {

    }

    public static TaskManager getInstance() {
        if (sInstance == null) {
            synchronized (TaskManager.class) {
                if (sInstance == null) {
                    sInstance = new TaskManager();
                }
            }
        }
        return sInstance;
    }

    public Looper taskLooper() {
        if (mTaskHandlerThread == null) {
            synchronized (TaskManager.class) {
                if (mTaskHandlerThread == null) {
                    mTaskHandlerThread = new HandlerThread("task handler thread");
                    mTaskHandlerThread.start();
                }
            }
        }
        return mTaskHandlerThread.getLooper();
    }

    public Looper backgroundLooper() {
        if (mBackgroudHandlerThread == null) {
            synchronized (TaskManager.class) {
                if (mBackgroudHandlerThread == null) {
                    mBackgroudHandlerThread = new HandlerThread("background handler thread");
                    mBackgroudHandlerThread.start();
                }
            }
        }
        return mBackgroudHandlerThread.getLooper();
    }
}
