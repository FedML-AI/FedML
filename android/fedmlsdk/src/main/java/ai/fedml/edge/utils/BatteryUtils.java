package ai.fedml.edge.utils;

import android.annotation.SuppressLint;
import android.content.Context;
import android.content.Intent;
import android.content.IntentFilter;
import android.os.BatteryManager;

import ai.fedml.edge.utils.entity.Battery;

public class BatteryUtils {
    public static Battery getBattery(Context context) {
        Battery battery = new Battery();
        try {
            Intent batteryStatus = context.registerReceiver(null,
                    new IntentFilter(Intent.ACTION_BATTERY_CHANGED));
            if (batteryStatus != null) {
                int level = batteryStatus.getIntExtra(BatteryManager.EXTRA_LEVEL, -1);
                int scale = batteryStatus.getIntExtra(BatteryManager.EXTRA_SCALE, -1);
                double batteryLevel = -1;
                if (level != -1 && scale != -1) {
                    batteryLevel = DoubleUtil.divide((double) level, (double) scale);
                }
                int status = batteryStatus.getIntExtra(BatteryManager.EXTRA_STATUS, -1);
                int health = batteryStatus.getIntExtra(BatteryManager.EXTRA_HEALTH, -1);
                battery.setPercentage(DoubleUtil.mul(batteryLevel, 100d) + "%");
                battery.setStatus(batteryStatus(status));
                battery.setHealth(batteryHealth(health));
                battery.setPower(getBatteryCapacity(context));
            }
        } catch (Exception e) {
            LogHelper.w(e, "getBattery failed.");
        }
        return battery;
    }

    @SuppressLint("PrivateApi")
    private static String getBatteryCapacity(Context context) {
        Object mPowerProfile;
        double batteryCapacity = 0;
        final String powerProfileClass = "com.android.internal.os.PowerProfile";
        try {
            mPowerProfile = Class.forName(powerProfileClass)
                    .getConstructor(Context.class)
                    .newInstance(context);

            batteryCapacity = (double) Class.forName(powerProfileClass)
                    .getMethod("getBatteryCapacity")
                    .invoke(mPowerProfile);
        } catch (Exception e) {
            LogHelper.w(e, "getBatteryCapacity failed.");
        }
        // unit is mAh
        return batteryCapacity + "";
    }

    private static String batteryHealth(int health) {
        String batteryhealth = "unknown";
        switch (health) {
            case BatteryManager.BATTERY_HEALTH_COLD:
                batteryhealth = "cold";
                break;
            case BatteryManager.BATTERY_HEALTH_DEAD:
                batteryhealth = "dead";
                break;
            case BatteryManager.BATTERY_HEALTH_GOOD:
                batteryhealth = "good";
                break;
            case BatteryManager.BATTERY_HEALTH_OVER_VOLTAGE:
                batteryhealth = "overVoltage";
                break;
            case BatteryManager.BATTERY_HEALTH_OVERHEAT:
                batteryhealth = "overheat";
                break;
            case BatteryManager.BATTERY_HEALTH_UNKNOWN:
                // Battery health is already set to unknown, so no need to do anything here
                break;
            case BatteryManager.BATTERY_HEALTH_UNSPECIFIED_FAILURE:
                batteryhealth = "unspecified";
                break;
            default:
                break;
        }
        return batteryhealth;
    }

    private static String batteryStatus(int status) {
        String batteryStatus = "unknown";
        switch (status) {
            case BatteryManager.BATTERY_STATUS_CHARGING:
                batteryStatus = "charging";
                break;
            case BatteryManager.BATTERY_STATUS_DISCHARGING:
                batteryStatus = "disCharging";
                break;
            case BatteryManager.BATTERY_STATUS_FULL:
                batteryStatus = "full";
                break;
            case BatteryManager.BATTERY_STATUS_NOT_CHARGING:
                batteryStatus = "notCharging";
                break;
            case BatteryManager.BATTERY_STATUS_UNKNOWN:
                // Battery status is already set to unknown, so no need to do anything here
                break;
            default:
                break;
        }
        return batteryStatus;
    }
}
