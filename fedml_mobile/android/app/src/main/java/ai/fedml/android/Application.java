package ai.fedml.android;


import android.content.BroadcastReceiver;
import android.content.Context;
import android.content.Intent;
import android.content.IntentFilter;
import android.net.ConnectivityManager;
import android.net.NetworkInfo;

import ai.fedml.iot.IOTCore;
import ai.fedml.iot.MyUncaughtExceptionHandler;
import ai.fedml.iot.utils.LogUtils;

public class Application extends android.app.Application {

    private static final String TAG = Application.class.getSimpleName();

    @Override
    protected void attachBaseContext(Context base) {
        super.attachBaseContext(base);
    }

    @Override
    public void onCreate() {
        super.onCreate();
        IOTCore.getInstance().init(this);

        registerBroadcast();

        //catch exception
        Thread.setDefaultUncaughtExceptionHandler(new MyUncaughtExceptionHandler(this));
    }

    @Override
    public void onTerminate() {
        super.onTerminate();
        unRegisterBroadcast();

        IOTCore.getInstance().unInit();
    }

    private void registerBroadcast() {
        IntentFilter filter = new IntentFilter();
        filter.addAction(ConnectivityManager.CONNECTIVITY_ACTION);
        filter.addAction(Intent.ACTION_SCREEN_ON);
        filter.addAction(Intent.ACTION_SCREEN_OFF);
        filter.addAction(Intent.ACTION_BOOT_COMPLETED);
        filter.addAction(Intent.ACTION_POWER_CONNECTED);
        filter.addAction(Intent.ACTION_POWER_DISCONNECTED);
        this.registerReceiver(mReceiver, filter);
    }

    private void unRegisterBroadcast() {
        try {
            this.unregisterReceiver(mReceiver);
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    private BroadcastReceiver mReceiver = new BroadcastReceiver() {
        @Override
        public void onReceive(Context context, Intent intent) {
            LogUtils.d(TAG, "ACTION = " + intent.getAction());
            if (Intent.ACTION_SCREEN_OFF.equals(intent.getAction())) {

            } else if (Intent.ACTION_SCREEN_ON.equals(intent.getAction())) {

            } else if (ConnectivityManager.CONNECTIVITY_ACTION.equals(intent.getAction())) {
                ConnectivityManager connectivityManager = (ConnectivityManager) context.getSystemService(Context.CONNECTIVITY_SERVICE);
                NetworkInfo info = connectivityManager.getActiveNetworkInfo();
                if (info != null && info.isAvailable()) {
                    onNetworkChanged(true, info.getType());
                } else {
                    onNetworkChanged(false, -1);
                }
            }
        }
    };

    private void onNetworkChanged(boolean isConnected, int networkType) {
        LogUtils.d(TAG, "onNetworkChanged. isConnected = " + isConnected);
        if (isConnected) {
            // do nothing
        }
    }
}