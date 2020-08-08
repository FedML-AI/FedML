package ai.fedml.iot.device;

import android.annotation.SuppressLint;
import android.content.Context;
import android.content.pm.PackageInfo;
import android.content.pm.PackageManager;
import android.net.wifi.WifiInfo;
import android.net.wifi.WifiManager;
import android.os.Build;
import android.telephony.TelephonyManager;
import android.text.TextUtils;
import android.util.Log;

import ai.fedml.iot.utils.LogUtils;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.IOException;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;

public class DeviceUtil {
    private static final String TAG = DeviceUtil.class.getSimpleName();

    private static String sAppVersionName;
    private static String sPackageName;
    private static int sAppVersionCode;

    private static String mMacAddr = "";

    private final static List<String> cidPaths = Collections.unmodifiableList(Arrays.asList(
            "/sys/block/mmcblk0/device/cid",
            "/sys/class/mmc_host/mmc0/mmc0:0001/cid",
            "/sys/devices/mtk-msdc.0/11230000.msdc0/mmc_host/mmc0/mmc0:0001/cid",
            "/sys/devices/platform/atc-msdc.0/mmc_host/mmc0/mmc0:0001/cid",
            "/sys/devices/platform/mmci-omap-hs.0/mmc_host/mmc0/mmc0:aaaa/cid",
            "/sys/devices/soc/7824900.sdhci/mmc_host/mmc0/mmc0:0001/cid",
            "/sys/devices/platform/emmc/mmc_host/mmc0/mmc0:0001/cid",
            "/sys/class/mmc_host/mmc0/mmc0:0001/serial",
            "/sys/devices/soc/7824900.sdhci/mmc_host/mmc0/mmc0:0001/serial",
            "/sys/devices/platform/emmc/mmc_host/mmc0/mmc0:0001/serial"
    ));

    public static String getCID() {
        String cid = "";
        for (String path : cidPaths) {
            String tmpId = getCidByPath(path);

            if (!TextUtils.isEmpty(tmpId)) {
                LogUtils.d(TAG, "path = " + path + ", cid = " + tmpId);
                cid = tmpId;
                break;
            }
        }
        if (TextUtils.isEmpty(cid)) {
            cid = getSimpleCID();
            LogUtils.d(TAG, "simple cid = " + cid);
        }
        return cid;
    }

    private static String getSimpleCID() {
        String cid = "";
        File input = new File("/sys/class/mmc_host/mmc0");
        String cid_directory = null;
        File[] sid = input.listFiles();
        if (sid != null && sid.length != 0) {
            for (int i = 0; i < sid.length; i++) {
                if (sid[i].toString().contains("mmc0:")) {
                    cid_directory = sid[i].toString();
                    break;
                }
            }
        }
        if (TextUtils.isEmpty(cid_directory)) {
            return cid;
        }
        BufferedReader cidBufferReader = null;
        try {
            cidBufferReader = new BufferedReader(new FileReader(
                    cid_directory + "/cid"));
            cid = cidBufferReader.readLine();
        } catch (Exception e) {
            e.printStackTrace();
        } finally {
            try {
                cidBufferReader.close();
            } catch (IOException e1) {
                e1.printStackTrace();
            }
        }
        return cid;
    }

    private static String getCidByPath(String path) {
        String sn = "";
        if (TextUtils.isEmpty(sn)) {
            BufferedReader snBufferReader = null;
            try {
                File snFile = new File(path);
                snBufferReader = new BufferedReader(new FileReader(snFile));
                sn = snBufferReader.readLine();
                Log.d("getCID", "getCID sn: " + sn);
            } catch (Exception e1) {
                //e1.printStackTrace();
                LogUtils.e(TAG, e1.getMessage());
                sn = "";
            } finally {
                try {
                    if (snBufferReader != null) {
                        snBufferReader.close();
                    }
                } catch (Exception e) {
                    //e.printStackTrace();
                    LogUtils.e(TAG, e.getMessage());
                }
            }
        }
        if (sn == null) {
            sn = "";
        }
        return sn;
    }

    @SuppressLint("MissingPermission")
    public static String getImeiNum(Context context) {

        String IMEI = "";
        try {
            TelephonyManager manager = (TelephonyManager) context.getSystemService(Context.TELEPHONY_SERVICE);
            if (manager.getDeviceId() == null || manager.getDeviceId().equals("")) {
                if (Build.VERSION.SDK_INT >= 23) {
                    IMEI = manager.getDeviceId(0);
                }
            } else {
                IMEI = manager.getDeviceId();
            }
        }catch (Exception e){
            e.printStackTrace();
        }
        return IMEI;
    }

    public static String getWifiMacAddress(Context context) {
        if (!TextUtils.isEmpty(mMacAddr)) {
            return mMacAddr;
        }
        String macAddress = null;

        WifiManager wifiManager = (WifiManager) context.getSystemService(Context.WIFI_SERVICE);
        macAddress = tryGetMAC(wifiManager);
        if (!TextUtils.isEmpty(macAddress)) {
            return macAddress;
        }

        boolean isTurnOnWifi = false;
        int state = wifiManager.getWifiState();
        if (state != WifiManager.WIFI_STATE_ENABLED && state != WifiManager.WIFI_STATE_ENABLING) {
            wifiManager.setWifiEnabled(true);
            isTurnOnWifi = true;
        }
        for (int index = 0; index < 50; index++) {
            //如果第一次没有成功，每过100ms查询一次，至多等待5s。
            if (index != 0) {
                try {
                    Thread.sleep(100);
                } catch (InterruptedException e) {
                    e.printStackTrace();
                }
            }
            macAddress = tryGetMAC(wifiManager);
            if (!TextUtils.isEmpty(macAddress)) {
                break;
            }
        }

        if (isTurnOnWifi) {
            wifiManager.setWifiEnabled(false);
        }

        return macAddress;
    }

    private static String tryGetMAC(WifiManager manager) {
        if (null == manager) {
            return null;
        }

        WifiInfo wifiInfo = manager.getConnectionInfo();
        if (wifiInfo == null || TextUtils.isEmpty(wifiInfo.getMacAddress())) {
            return null;
        }
        String mac = wifiInfo.getMacAddress().trim().toUpperCase();
        return mac;
    }

    public static String getBrand() {
        String brand = Build.BRAND;
        if (brand == null) {
            brand = "";
        }
        return brand;
    }

    public static String getManufacturer() {
        String manufacturer = Build.MANUFACTURER;
        if (manufacturer == null) {
            manufacturer = "";
        }
        return manufacturer;
    }

    public static String getModel() {
        String model = Build.MODEL;
        if (model == null) {
            model = "";
        }
        return model;
    }

    public static String getProduct(){
        String product = Build.PRODUCT;
        if (product == null) {
            product = "";
        }
        return product;
    }

    public static String getHardware(){
        String hardware = Build.HARDWARE;
        if (hardware == null) {
            hardware = "";
        }
        return hardware;
    }

    public static String getBoard(){
        String board = Build.BOARD;
        if (board == null) {
            board = "";
        }
        return board;
    }

    public static String getDevice(){
        String device = Build.DEVICE;
        if (device == null) {
            device = "";
        }
        return device;
    }

    public static String getOSVersion() {
        return Build.VERSION.RELEASE;
    }

    public static int getAppVersionCode(Context context) {
        initAppVersion(context);
        return sAppVersionCode;
    }

    public static String getAppVersionName(Context context) {
        initAppVersion(context);
        return sAppVersionName;
    }

    public static String getAppPackageName(Context context) {
        initAppVersion(context);
        return sPackageName;
    }

    private static void initAppVersion(Context context) {
        if (!TextUtils.isEmpty(sPackageName)) {
            return;
        }
        PackageManager manager = context.getPackageManager();
        try {
            sPackageName = context.getPackageName();
            PackageInfo info = manager.getPackageInfo(sPackageName, 0);
            sAppVersionName = info.versionName;
            sAppVersionCode = info.versionCode;
        } catch (PackageManager.NameNotFoundException e) {
            sAppVersionName = "NULL";
            sAppVersionCode = -1;
        }
    }


}
