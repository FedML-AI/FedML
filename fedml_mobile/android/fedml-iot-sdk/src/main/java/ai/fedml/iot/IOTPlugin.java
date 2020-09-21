package ai.fedml.iot;

import android.app.Application;
import android.os.Bundle;

import ai.fedml.iot.utils.LogUtils;

public class IOTPlugin {
    private String TAG = Config.COMMON_TAG + getClass().getSimpleName();

    IOTPluginImpl mIOVPluginImpl = null;

    public int processMessage(Bundle args) {
        return 0;
    }

    public void init(Application applicationContext, String vehicleID, String channelID,
                     String pluginPkgName, String pluginVersionName, int pluginVersionCode) {
        ApplicationContextHolder.setContext(applicationContext);

        LogUtils.setLogEnable(false);

        mIOVPluginImpl = new IOTPluginImpl(applicationContext);
        mIOVPluginImpl.init();

        LogUtils.e(TAG, "IOVPlugin INIT END");

    }

    public void uninit() {
        LogUtils.e(TAG, "uninit");
        if(mIOVPluginImpl != null){
            mIOVPluginImpl.unInit();
            mIOVPluginImpl = null;
        }
    }
}
