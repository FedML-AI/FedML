package ai.fedml.iot;

import android.app.Application;
import android.content.BroadcastReceiver;
import android.content.Context;
import android.content.Intent;
import android.content.IntentFilter;

import ai.fedml.iot.service.IOTServiceServiceManager;
import ai.fedml.iot.utils.LogUtils;

public class IOTCore {
    private static final String TAG = IOTCore.class.getSimpleName();

    private static final IOTCore ourInstance = new IOTCore();

    public static IOTCore getInstance() {
        return ourInstance;
    }

    private boolean mInitialized = false;
    private Application mApplication;
    private Context mContext;

    public synchronized void init(Application application) {
        if (mInitialized) return;

        mApplication = application;
        mContext = application.getApplicationContext();

        //catch exception
        Thread.setDefaultUncaughtExceptionHandler(new MyUncaughtExceptionHandler(mApplication));

        //open log
        LogUtils.setLogEnable(true);

        //init process keeping alive service
        mContext.startService(new Intent(mContext, DaemonService.class));

        //init MQ service. Inside the MQ service, it triggers location SDK and start to collect trajectory.
        IOTServiceServiceManager.getInstance().init(getContext());

        registerBroadcast();

        mInitialized = true;
    }

    public Context getContext() {
        if (mContext == null) {
            return null;
        }
        return mContext;
    }

    public void unInit(){
        IOTServiceServiceManager.getInstance().destroy();
        unRegisterBroadcast();
    }


    private void registerBroadcast() {
        IntentFilter filter = new IntentFilter();
        filter.addAction(Config.BROADCAST_RELOAD_PLUGIN);
        mContext.registerReceiver(mSystemBroadcastReceiver, filter);
    }

    private void unRegisterBroadcast() {
        try {
            mContext.unregisterReceiver(mSystemBroadcastReceiver);
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    private BroadcastReceiver mSystemBroadcastReceiver = new BroadcastReceiver() {
        @Override
        public void onReceive(Context context, Intent intent) {
            if (Config.BROADCAST_RELOAD_PLUGIN.equals(intent.getAction())) {
                IOTServiceServiceManager.getInstance().unbindService();
            }
        }
    };
}
