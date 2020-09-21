package ai.fedml.iot.utils;

import android.text.TextUtils;

import java.lang.reflect.Method;

public class SystemProperty {
    public static String get(String key) {
        if (TextUtils.isEmpty(key)) {
            return null;
        }
        String value = "";
        try {
            Method get = null;
            synchronized (SystemProperty.class) {
                Class<?> cls = Class.forName("android.os.SystemProperties");
                get = cls.getDeclaredMethod("get", new Class<?>[]{String.class, String.class});
            }
            value = (String) (get.invoke(null, new Object[]{key, ""}));
        } catch (Throwable e) {
            e.printStackTrace();
        }
        return value;
    }
}
