package ai.fedml.iot;

import android.content.BroadcastReceiver;
import android.content.Context;
import android.content.Intent;
import android.content.IntentFilter;
import android.net.ConnectivityManager;
import android.net.NetworkInfo;

import ai.fedml.iot.flmsg.FedMLMessenger;
import ai.fedml.iot.mq.IMQListener;
import ai.fedml.iot.mq.MQManager;
import ai.fedml.iot.utils.LogUtils;

import org.fusesource.hawtbuf.Buffer;
import org.fusesource.hawtbuf.UTF8Buffer;

public class IOTPluginImpl {
    private String TAG = Config.COMMON_TAG + getClass().getSimpleName();

    private Context mContext;
    private MQManager mMQManager = null;

    public IOTPluginImpl(Context context) {
        mContext = context;
    }

    public void init() {
        LogUtils.e(TAG, "initial app domain = " + Config.BASE_URL_IOT_APP_SERVICE);
        LogUtils.e(TAG, "initial data domain = " + Config.BASE_URL_IOT_MQ_SERVICE);

        //init MQTT
        mMQManager = new MQManager();
        try {
            mMQManager.connect(mMQListener);
        } catch (Exception e) {
            e.printStackTrace();
        }

        FedMLMessenger.getInstance().init(mContext, mMQManager);

        //register broadcast
        registerBroadcast();
    }

    public void unInit() {
        unRegisterBroadcast();

        FedMLMessenger.destroy();

        if (mMQManager != null) {
            mMQManager.release();
        }
    }

    private void registerBroadcast() {
        IntentFilter filter = new IntentFilter();
        filter.addAction(ConnectivityManager.CONNECTIVITY_ACTION);
        //filter.addAction("android.net.conn.CONNECTIVITY_CHANGE");
        filter.addAction(Intent.ACTION_SCREEN_ON);
        filter.addAction(Intent.ACTION_SCREEN_OFF);
        filter.addAction(Intent.ACTION_BOOT_COMPLETED);
        filter.addAction(Intent.ACTION_POWER_CONNECTED);
        filter.addAction(Intent.ACTION_POWER_DISCONNECTED);
        mContext.registerReceiver(mSystemBroadcastReceiver, filter);
    }

    private void unRegisterBroadcast() {
        try {
            mContext.unregisterReceiver(mSystemBroadcastReceiver);
        } catch (Exception e) {
            e.printStackTrace();
        }
        LogUtils.e(TAG, "unRegisterBroadcast");
    }

    //SCREEN_ON -> ACTION_POWER_CONNECTED -> CONNECTIVITY_CHANGE(true)
    //CONNECTIVITY_CHANGE(true) -> CONNECTIVITY_CHANGE(false) -> SCREEN_OFF
    private BroadcastReceiver mSystemBroadcastReceiver = new BroadcastReceiver() {
        @Override
        public void onReceive(Context context, Intent intent) {
            LogUtils.d(TAG, "ACTION = " + intent.getAction());
            if (Intent.ACTION_SCREEN_OFF.equals(intent.getAction())) {

            } else if (Intent.ACTION_SCREEN_ON.equals(intent.getAction())) {

                networking();

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
            networking();
        }
    }

    private void networking() {
        reconnectMQTT();
    }

    private void reconnectMQTT() {
        LogUtils.e(TAG, "reconnectMQTT");
        if(mMQManager != null && mMQManager.getState() == MQManager.CONNECT_STATE_CONNECTED
                || mMQManager.getState() == MQManager.CONNECT_STATE_CONNECTING){
            return ;
        }
        if(mMQManager == null){
            mMQManager = new MQManager();
        }else{
            mMQManager.release();
        }
        try {
            LogUtils.e(TAG, "reconnectMQTT");
            mMQManager.connect(mMQListener);
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    private IMQListener mMQListener = new IMQListener() {
        @Override
        public void onConnected() {
        }

        @Override
        public void onDisconnected() {

        }

        @Override
        public void onFailure(Throwable throwable) {

        }

        @Override
        public void onPublish(UTF8Buffer utf8Buffer, Buffer buffer, Runnable runnable) {

        }
    };

}