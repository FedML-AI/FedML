package ai.fedml.edge.utils;

import android.content.Context;
import android.provider.Settings;
import android.text.TextUtils;
import android.util.Log;
import android.os.Process;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.IOException;

import ai.fedml.edge.service.ContextHolder;

/**
 * Used to obtain the unique identity of the device
 */
public class DeviceUtils {
    private static final String TAG = "DeviceUtils";
    private static String sDeviceId;

    private DeviceUtils() {
    }

    public static String getDeviceId() {
        if (TextUtils.isEmpty(sDeviceId)) {
            sDeviceId = getUniqueId(ContextHolder.getAppContext());
        }
        return sDeviceId;
    }

    public static String getUniqueId(Context context) {
        return Settings.System.getString(context.getContentResolver(), Settings.Secure.ANDROID_ID);
    }

    public static String getProcessName() {
        final File file = new File("/proc/" + Process.myPid() + "/cmdline");
        try (BufferedReader mBufferedReader = new BufferedReader(new FileReader(file))) {
            return mBufferedReader.readLine().trim();
        } catch (IOException e) {
            Log.e(TAG, "getProcessName", e);
        }
        return null;
    }
}
