package ai.fedml.iot;

import android.content.BroadcastReceiver;
import android.content.Context;
import android.content.Intent;

import ai.fedml.iot.utils.LogUtils;


public class AutoStartReceiver extends BroadcastReceiver {
    private static final String TAG = AutoStartReceiver.class.getSimpleName();

    @Override
    public void onReceive(Context context, Intent intent) {
        LogUtils.d(TAG, "ACTION = " + intent.getAction() );
    }
}
