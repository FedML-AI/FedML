package ai.fedml.iot.utils;

import android.app.ActivityManager;
import android.content.ComponentName;
import android.content.Context;
import android.content.ContextWrapper;
import android.content.Intent;
import android.content.pm.PackageManager;
import android.content.pm.ResolveInfo;
import android.os.UserHandle;

import java.lang.reflect.Method;
import java.util.List;

public class ServiceUtils {

    /**
     * Get Service Explicit Intent
     * @param context
     * @param implicitIntent
     * @return
     */
    public static Intent getExplicitIntent(Context context, Intent implicitIntent) {
        // Retrieve all services that can match the given intent
        PackageManager pm = context.getPackageManager();
        List<ResolveInfo> resolveInfo;
        try {
            resolveInfo = pm.queryIntentServices(implicitIntent, 0);
        } catch (Exception ex) {
            // bugly:android.os.DeadObjectException:
            LogUtils.e("ServiceUtils", ex.toString());
            resolveInfo = null;
        }

        // Make sure only one match was found
        if (resolveInfo == null || resolveInfo.size() != 1) {
            return null;
        }
        // Get component info and create ComponentName
        ResolveInfo serviceInfo = resolveInfo.get(0);
        String packageName = serviceInfo.serviceInfo.packageName;
        String className = serviceInfo.serviceInfo.name;
        ComponentName component = new ComponentName(packageName, className);
        // Create a new intent. Use the old one for extras and such reuse
        Intent explicitIntent = new Intent(implicitIntent);
        // Set the component to be explicit
        explicitIntent.setComponent(component);
        return explicitIntent;
    }

    /**
     * @param context
     * @param implicitIntent
     * @param pkg
     * @return
     */
    public static Intent getMatchIntentInPackage(Context context, Intent implicitIntent, String pkg) {
        // Retrieve all services that can match the given intent
        PackageManager pm = context.getPackageManager();
        List<ResolveInfo> resolveInfo;
        try {
            implicitIntent.addFlags(Intent. FLAG_INCLUDE_STOPPED_PACKAGES);
            resolveInfo = pm.queryIntentServices(implicitIntent, 0);
        } catch (Exception ex) {
            // bugly:android.os.DeadObjectException:
            LogUtils.e("ServiceUtils", ex.toString());
            resolveInfo = null;
        }

        // Make sure only one match was found
        if (resolveInfo == null || resolveInfo.size() == 0) {
            return null;
        }
        Intent explicitIntent = null;
        for (int i = 0; i < resolveInfo.size(); i++) {
            // Get component info and create ComponentName
            ResolveInfo serviceInfo = resolveInfo.get(i);
            if (serviceInfo.serviceInfo.packageName.equals(pkg)){
                String packageName = serviceInfo.serviceInfo.packageName;
                String className = serviceInfo.serviceInfo.name;
                ComponentName component = new ComponentName(packageName, className);
                // Create a new intent. Use the old one for extras and such reuse
                explicitIntent = new Intent(implicitIntent);
                // Set the component to be explicit
                explicitIntent.setComponent(component);
                break;
            }
        }
        return explicitIntent;
    }

    /**
     * @param context
     * @param action
     * @return
     */
    public static boolean isServiceRunning(Context context, String action) {
        ActivityManager am = (ActivityManager) context.getSystemService(Context.ACTIVITY_SERVICE);
        List<ActivityManager.RunningServiceInfo> serviceList = am.getRunningServices(Integer.MAX_VALUE);
        if (serviceList != null) {
            for (ActivityManager.RunningServiceInfo serviceInfo : serviceList) {
                if (action.equals(serviceInfo.service.getClassName())) {
                    return true;
                }
            }
        }
        return false;
    }

    public static boolean isServiceRunning(Context context, String pkgName, Class<?> serviceClass) {
        ActivityManager am = (ActivityManager) context.getSystemService(Context.ACTIVITY_SERVICE);
        List<ActivityManager.RunningServiceInfo> serviceList = am.getRunningServices(Integer.MAX_VALUE);
        if (serviceList != null) {
            for (ActivityManager.RunningServiceInfo serviceInfo : serviceList) {
                if (serviceInfo.service.getPackageName().equals(pkgName)
                        && serviceInfo.service.getClassName().equals(serviceClass.getName())) {
                    return true;
                }
            }
        }
        return false;
    }

    /**
     * @param action
     */
    public static void startService(Context context, String action) {
        Intent intent = getExplicitIntent(context, new Intent(action));
        if (intent != null) {
            if (!isServiceRunning(context, action)) {
//			LogUtils.d(TAG, "service is not running, start it");
                Intent serviceIntent = new Intent(action);
                try {
                    Class[] argsClass = new Class[2];
                    argsClass[0] = Intent.class;
                    argsClass[1] = UserHandle.class;
                    ContextWrapper contextWrapper = new ContextWrapper(context);
                    Method method = contextWrapper.getClass().getMethod("startServiceAsUser", argsClass);
                    method.invoke(contextWrapper, getExplicitIntent(context, serviceIntent), android.os.Process.myUserHandle());
                } catch (Exception e) {
                    e.printStackTrace();
                }
            } else {
//            LogUtils.d(TAG, "service is already running, no need to start it again");
            }
        } else {
//            LogUtils.d(TAG, "startService failed, action = " + action);
        }
    }
}
