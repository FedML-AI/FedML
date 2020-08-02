package ai.fedml.iot.service;

import android.app.Service;
import android.content.Intent;
import android.os.IBinder;
import android.util.Log;

import ai.fedml.iot.utils.LogUtils;

import static ai.fedml.iot.CommonConfig.INTENT_KEY_BUNDLE;


public class IOTService extends Service {

    private String TAG = "install_" + getClass().getSimpleName();

    public static final String ACTION_MQTT_SERVICE = "ai.fedml.iot.MQTT_SERVICE";

    IOTServiceImpl mService;

    @Override
    public void onCreate() {
        super.onCreate();
        LogUtils.e(TAG, "onCreate");
        mService = new IOTServiceImpl(getApplication());
        mService.init();
    }

    @Override
    public IBinder onBind(Intent intent) {
        String action = intent.getAction();
        LogUtils.e(TAG, "onBind, action = " + action);
        if (ACTION_MQTT_SERVICE.equals(action)) {
            LogUtils.d(TAG, "onBind, mService = " + mService);
            return mService;
        }
        LogUtils.d(TAG, "onBind, return null");
        return null;
    }

    @Override
    public int onStartCommand(Intent intent, int flags, int startId) {
        try {
            Log.d(TAG, TAG + " onStartCommand " + intent);
            super.onStartCommand(intent, flags, startId);
            if (intent == null || !intent.hasExtra(INTENT_KEY_BUNDLE) ||
                    intent.getBundleExtra(INTENT_KEY_BUNDLE) == null) {
                return START_REDELIVER_INTENT;
            }
        } catch (Exception e) {
            e.printStackTrace();
            return START_REDELIVER_INTENT;
        }

        return mService.onStartCommand(intent);
    }

    @Override
    public boolean onUnbind(Intent intent) {
        LogUtils.e(TAG, "onUnbind");
        if (mService != null) {
            mService.unInit();
        }
        return super.onUnbind(intent);
    }

    @Override
    public void onDestroy() {
        super.onDestroy();
        LogUtils.e(TAG, "onDestroy");
        if (mService != null) {
            mService.unInit();
            mService = null;
        }
    }
}
