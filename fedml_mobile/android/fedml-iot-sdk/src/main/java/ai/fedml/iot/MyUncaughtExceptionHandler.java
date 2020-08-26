package ai.fedml.iot;

import android.app.ActivityManager;
import android.app.Application;
import android.content.Context;
import android.os.Looper;
import android.widget.Toast;

import ai.fedml.iot.utils.LogUtils;

public class MyUncaughtExceptionHandler implements Thread.UncaughtExceptionHandler {

    private static final String TAG = MyUncaughtExceptionHandler.class.getSimpleName();
    private Thread.UncaughtExceptionHandler mUncaughtExceptionHandler;
    private Application mApplication;
    private boolean bShowLog = false;

    private static final boolean bDebugMode = false;

    public MyUncaughtExceptionHandler(Application context) {
        mApplication = context;
        mUncaughtExceptionHandler = Thread.getDefaultUncaughtExceptionHandler();
    }

    @Override
    public void uncaughtException(Thread thread, Throwable ex) {
        if (mApplication.getPackageName().equals(getProcessName(mApplication)))  {
            if (bShowLog) {
                LogUtils.e(TAG, "uncaughtException, show log");
                mUncaughtExceptionHandler.uncaughtException(thread, ex);
            } else {
                LogUtils.e(TAG, "toast Sorry");
                //toastSorry();
                restart();
                android.os.Process.killProcess(android.os.Process.myPid());
            }
        } else {
            LogUtils.e(TAG, "kill process directly!");//here
            ex.printStackTrace();
            if(bDebugMode){
                toastSorry();
            }
            try {
                Thread.sleep(1000);
            } catch (InterruptedException e) {
                e.printStackTrace();
            }
            android.os.Process.killProcess(android.os.Process.myPid());
        }
    }

    private void toastSorry() {
        new Thread(){
            @Override
            public void run() {
                Looper.prepare();
                Toast.makeText(
                        mApplication.getApplicationContext(),
                        "Program is abnormal, is about to restart the application.",
                        Toast.LENGTH_SHORT).show();
                Looper.loop();
            }
        }.start();
    }

    public static String getProcessName(Context appContext) {
        String currentProcessName = "";
        int pid = android.os.Process.myPid();
        ActivityManager manager = (ActivityManager) appContext.getSystemService(Context.ACTIVITY_SERVICE);
        for (ActivityManager.RunningAppProcessInfo processInfo : manager.getRunningAppProcesses()) {
            if (processInfo.pid == pid) {
                currentProcessName = processInfo.processName;
                break;
            }
        }
        return currentProcessName;
    }

    private void restart() {
        LogUtils.e(TAG, "restart");
//        try {
//            Thread.sleep(4000);
//        } catch (InterruptedException e) {
//            android.util.Log.e(TAG, "error : " + e);
//        }
//        Intent intent = new Intent(
//                mApplication.getApplicationContext(),
//                MainActivity.class);
//        @SuppressLint("WrongConstant") PendingIntent restartIntent = PendingIntent.getActivity(
//                mApplication.getApplicationContext(), 0, intent,
//                Intent.FLAG_ACTIVITY_NEW_TASK);

//        AlarmManager mgr = (AlarmManager) mApplication.getSystemService(Context.ALARM_SERVICE);
//        mgr.set(AlarmManager.RTC, System.currentTimeMillis() + 1000,
//                restartIntent);

    }
}