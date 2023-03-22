package ai.fedml.edge.service;

import android.annotation.SuppressLint;
import android.app.Application;
import android.content.Context;
import android.util.Log;

public class ContextHolder {

    private static final String ANDROID_APP_ACTIVITY_THREAD = "android.app.ActivityThread";
    private static final String CURRENT_APPLICATION = "currentApplication";
    public static final String ANDROID_APP_APP_GLOBALS = "android.app.AppGlobals";
    public static final String GET_INITIAL_APPLICATION = "getInitialApplication";
    @SuppressLint("StaticFieldLeak")
    private static Context sContext;

    private ContextHolder() {
    }

    @SuppressLint("PrivateApi")
    private static Application getApplicationUsingReflection() throws Exception {
        return (Application) Class.forName(ANDROID_APP_ACTIVITY_THREAD)
                .getMethod(CURRENT_APPLICATION).invoke(null, (Object[]) null);
    }

    @SuppressLint("PrivateApi")
    private static Application getApplicationUsingAppGlobalsReflection() throws Exception {
        return (Application) Class.forName(ANDROID_APP_APP_GLOBALS)
                .getMethod(GET_INITIAL_APPLICATION).invoke(null, (Object[]) null);
    }

    /**
     * init Application Context
     *
     * @param context Context from Application
     */
    public static void initialize(Context context) {
        if (context == null) return;
        sContext = context.getApplicationContext();
    }

    public static Context getAppContext() {
        if (sContext != null) {
            return sContext;
        }
        try {
            sContext = getApplicationUsingAppGlobalsReflection().getApplicationContext();
        } catch (Exception e) {
            Log.e("ContextHolder", "getApplicationUsingAppGlobalsReflection failed! ", e);
        }
        if (sContext != null) {
            return sContext;
        }
        try {
            sContext = getApplicationUsingReflection().getApplicationContext();
        } catch (Exception e) {
            Log.e("ContextHolder", "getApplicationUsingReflection failed! ", e);
        }
        if (sContext != null) {
            return sContext;
        }
        throw new RuntimeException("ContextHolder is null! Please initialize it!");
    }
}
