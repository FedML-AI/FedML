package ai.fedml.iot.service;

import android.annotation.SuppressLint;
import android.app.Application;
import android.content.Intent;
import android.os.Build;
import android.os.Bundle;
import android.os.Handler;
import android.os.Message;
import android.os.RemoteException;

import ai.fedml.iot.Config;
import ai.fedml.iot.IMQService;
import ai.fedml.iot.IOTPlugin;
import ai.fedml.iot.device.IoTDevice;
import ai.fedml.iot.utils.BackgroundHandler;
import ai.fedml.iot.utils.LogUtils;

import static android.app.Service.START_REDELIVER_INTENT;
import static ai.fedml.iot.CommonConfig.BUNDLE_KEY_COMMAND;

public class IOTServiceImpl extends IMQService.Stub {
    private String TAG = "install_" + getClass().getSimpleName();
    public static final String INTENT_KEY_BUNDLE = "bundle";

    private IOTPlugin servicePlugin = null;
    private boolean initialized = false;

    private Application application = null;

    public IOTServiceImpl(Application app) {
        application = app;
    }

    public void init() {
        //install plugin framework
        LogUtils.e(TAG, "init");

        IoTDevice.init(application);
    }

    public void unInit() {
        LogUtils.e(TAG, "unInit");
        if(servicePlugin!=null) {
            servicePlugin.uninit();
            servicePlugin = null;
        }
        initialized = false;
    }

    public int onStartCommand(Intent intent){
        LogUtils.e(TAG, "onStartCommand");
        Bundle bundle = intent.getBundleExtra(INTENT_KEY_BUNDLE);
        mHandler.obtainMessage(SERVICE_PROCESS_MSG, bundle).sendToTarget();
        return START_REDELIVER_INTENT;
    }

    private static final int SERVICE_PROCESS_MSG = 0x1001;
    @SuppressLint("HandlerLeak")
    private Handler mHandler = new BackgroundHandler() {
        @Override
        public void handleMessage(Message msg) {
            super.handleMessage(msg);
            switch (msg.what) {
                case SERVICE_PROCESS_MSG:
                    try {
                        LogUtils.d(TAG,"handleMessage: SERVICE_PROCESS_MSG");
                        processInternal((Bundle) msg.obj);
                    } catch (Exception e) {
                        e.printStackTrace();
                    }
                    break;
            }
        }
    };

    private void processInternal(Bundle bundle) {
        LogUtils.d(TAG, "processInternal");

        if(!initialized && servicePlugin == null) {
            LogUtils.d(TAG, "processInternal init");

            IoTDevice.init(application);

            //start SDK initial
            Config.DeviceId = IoTDevice.getDeviceID();
            Config.Channel = IoTDevice.getChannelID();

            Config.AppPackageName = IoTDevice.getAppPackageName(application);
            Config.AppVersionName = IoTDevice.getAppVersionName(application);
            Config.AppVersionCode = IoTDevice.getAppVersionCode(application);
            Config.AndroidApi = Build.VERSION.SDK_INT;

            servicePlugin = new IOTPlugin();
            servicePlugin.init(application, Config.DeviceId, Config.Channel,
                    Config.PluginPackageName, Config.PluginVersionName, Config.PluginVersionCode);
            LogUtils.e(TAG, "plugin install successfully. servicePlugin = " + servicePlugin);
        }

        if(servicePlugin!=null) {
            servicePlugin.processMessage(bundle);
        }
    }

    @Override
    public int processMessage(int command) throws RemoteException {
        Config.DeviceId = IoTDevice.getDeviceID(application);
        Config.Channel = IoTDevice.getChannelID();

        Bundle msgBundle = new Bundle();
        msgBundle.putInt(BUNDLE_KEY_COMMAND, command);
        if(servicePlugin!=null) {
            servicePlugin.processMessage(msgBundle);
        }
        return 0;
    }

    @Override
    public void onServiceConnectedOk() throws RemoteException {
        LogUtils.e(TAG, "onServiceConnectedOk");
    }

    @Override
    public void onServiceDisconnected() throws RemoteException {
        LogUtils.e(TAG, "onServiceDisconnected");
        if(servicePlugin!=null) {
            servicePlugin.uninit();
            servicePlugin = null;
        }
        initialized = false;
    }
}