package ai.fedml.iot.utils;

import android.content.Context;
import android.util.DisplayMetrics;

import ai.fedml.iot.ApplicationContextHolder;

import java.lang.reflect.Field;

public class ScreenUtil {

    private static float mDensity = 0;
    private static int mStatusBarHeight = 0;
    private static int mDPI = 0;
    private static int mHeightPixels = 0;
    private static int mWidthPixels = 0;


    private static Context getContext() {
        return ApplicationContextHolder.getContext();
    }

    public static float getDensity() {
        if (mDensity == 0) {
            Context context = getContext();
            if (context != null) {
                DisplayMetrics displayMetrics = context.getResources().getDisplayMetrics();
                mDensity = displayMetrics.density;
                mDPI = displayMetrics.densityDpi;
                mHeightPixels = displayMetrics.heightPixels;
                mWidthPixels = displayMetrics.widthPixels;
            }
        }
        return mDensity;
    }

    public static boolean isHDPI() {
        if (getMapDensity() > 2) {
            return true;
        }
        return false;
    }

    public static float getMapDensity() {
        if (getDensity() < 1.5) {
            return 1.5f;
        }
        return getDensity();
    }


    public static int getDPI() {
        if (mDPI == 0) {
            Context context = getContext();
            if (context != null) {
                DisplayMetrics displayMetrics = context.getResources().getDisplayMetrics();
                mDensity = displayMetrics.density;
                mDPI = displayMetrics.densityDpi;
            }
        }
        return mDPI;
    }

    public static int getWidthPixels() {
        if (mWidthPixels == 0) {
            Context context = getContext();
            if (context != null) {
                DisplayMetrics displayMetrics = context.getResources().getDisplayMetrics();
                mWidthPixels = displayMetrics.widthPixels;
            }
        }
        return mWidthPixels;
    }

    public static int getHeightPixels() {
        if (mHeightPixels == 0) {
            Context context = getContext();
            if (context != null) {
                DisplayMetrics displayMetrics = context.getResources().getDisplayMetrics();
                mHeightPixels = displayMetrics.heightPixels;
            }
        }
        return mHeightPixels;
    }

    public static int getStatusBarHeight() {
        if (mStatusBarHeight == 0) {
            Context context = getContext();
            if (context != null) {
                try {
                    Class<?> c = Class.forName("com.android.internal.R$dimen");
                    Object obj = c.newInstance();
                    Field field = c.getField("status_bar_height");
                    int x = Integer.parseInt(field.get(obj).toString());
                    mStatusBarHeight = context.getResources().getDimensionPixelSize(x);
                } catch (Throwable e) {

                }
            }
        }
        return mStatusBarHeight;
    }


    public static int dip2px(int dip) {
        return (int) (0.5F + getDensity() * dip);
    }

    public static int px2dip(int px) {
        float dst = getDensity();
        dst = (dst == 0.0f) ? 1.0f : dst;
        return (int) (0.5F + px / dst);
    }

    public static int dip2px(float dip) {
        return (int) (0.5F + getDensity() * dip);
    }

    public static int px2dip(float px) {
        float dst = getDensity();
        dst = (dst == 0.0f) ? 1.0f : dst;
        return (int) (0.5F + px / dst);
    }

    public static int percentHeight(float percent) {
        return (int) (percent * getHeightPixels());
    }

    public static int percentWidth(float percent) {
        return (int) (percent * getWidthPixels());
    }


}
