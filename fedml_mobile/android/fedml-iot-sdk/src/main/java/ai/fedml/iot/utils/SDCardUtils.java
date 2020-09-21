package ai.fedml.iot.utils;

import android.os.Environment;
import android.text.TextUtils;

import ai.fedml.iot.Config;

import java.io.File;

public class SDCardUtils {
    public static final String TAG = SDCardUtils.class.getSimpleName();

    public static String sExternalPath;
    public static String sExternalDataPath;
    public static String mSdcardRootPath;



    public static String getSDcardRootPath() {
        if (TextUtils.isEmpty(mSdcardRootPath)) {
            mSdcardRootPath =  Environment.getExternalStorageDirectory().getAbsolutePath()+"/";
        }
        return mSdcardRootPath;
    }

    public synchronized  static String getSDcardPath() {
        String path = (sExternalPath == null ? Config.SDCARD_PATH : sExternalPath);
        File file = new File(getSDcardRootPath(),path);
        if (!file.exists()) {
            if (!file.mkdirs()) {
                LogUtils.e(SDCardUtils.class.getSimpleName(), "" +
                        "getSDcardPath mkdirs failed");
            }
        }
        return file.getAbsolutePath()+"/";
    }

    public synchronized  static String getSDcardTempPath() {
        File file = new File(getSDcardPath()+ Config.SDCARD_TEMP_PATH);
        if (!file.exists()) {
            if (!file.mkdirs()) {
                LogUtils.e(SDCardUtils.class.getSimpleName(), "" +
                        "getSDcardTempPath mkdirs failed");
            }
        }
        return file.getAbsolutePath()+"/";
    }

}
