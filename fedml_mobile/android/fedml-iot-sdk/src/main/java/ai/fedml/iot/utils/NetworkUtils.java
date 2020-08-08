package ai.fedml.iot.utils;

import android.content.Context;
import android.net.ConnectivityManager;
import android.net.NetworkInfo;
import android.net.wifi.WifiInfo;
import android.net.wifi.WifiManager;
import android.telephony.TelephonyManager;

import ai.fedml.iot.Config;

import java.util.HashMap;

public class NetworkUtils {
    private static final String TAG = Config.COMMON_TAG + NetworkUtils.class.getSimpleName();
    public static final int CONNECT_NONE = 0;
    public static final int CONNECT_MOBILE=1;
    public static final int CONNECT_WIFI=2;
    public static final int CONNECT_2G = 3;
    public static final int CONNECT_3G = 4;
    public static final int CONNECT_4G = 5;

    public static boolean isUsingWifiNetwork(Context context){

        ConnectivityManager connManager = (ConnectivityManager) context
                .getSystemService(Context.CONNECTIVITY_SERVICE);
        if(connManager == null)
            return false;
        NetworkInfo mWifi = connManager.getNetworkInfo(ConnectivityManager.TYPE_WIFI);
        if(mWifi == null)
            return false;
        return mWifi.isConnected();
    }

    public static boolean isUsingMobileNetwork(Context context){
        try{
            ConnectivityManager connec = (ConnectivityManager) context
                    .getSystemService(Context.CONNECTIVITY_SERVICE);
            NetworkInfo info = connec.getActiveNetworkInfo();
            String typeName = "";
            if (info != null) {
                typeName = info.getTypeName();
            } else {
                return false;
            }

            if ("mobile".equalsIgnoreCase(typeName)) {
                return true;
            } else {
                return false;
            }
        }catch(Throwable e){
            e.printStackTrace();
        }
        return false;
    }

    public static boolean isNetworkAvailable(Context context) {
        long time = System.currentTimeMillis();
        if (context == null)
            return false;
        try{
            ConnectivityManager connectivity = (ConnectivityManager) context
                    .getSystemService(Context.CONNECTIVITY_SERVICE);
            if (connectivity == null) {
                LogUtils.fe(TAG, "+++couldn't get connectivity manager");
            } else {
                NetworkInfo[] info = connectivity.getAllNetworkInfo();
                if (info != null) {
                    for (int i = 0; i < info.length; i++) {
                        if (info[i].getState() == NetworkInfo.State.CONNECTED) {
                            //Log.d(TAG, "+++network is available, cost time: " + (System.currentTimeMillis() - time));
                            return true;
                        }
                    }
                }
            }
        }catch(Throwable e){
            e.printStackTrace();
        }

        //Log.d(TAG, "+++network is not available, cost time: " + (System.currentTimeMillis() - time));
        return false;
    }

    public static int getConnectType(Context context) {
        ConnectivityManager connec = (ConnectivityManager) context
                .getSystemService(Context.CONNECTIVITY_SERVICE);
        if(connec == null){
            return CONNECT_NONE;
        }
        NetworkInfo info = connec.getActiveNetworkInfo();
        String typeName = "";
        int net = CONNECT_NONE;
        if (info != null) {
            typeName = info.getTypeName();
        } else {
            return net;
        }

        if ("mobile".equalsIgnoreCase(typeName)) {
            net = getMobileConnectType(context);
        } else if ("wifi".equalsIgnoreCase(typeName)) {
            net = CONNECT_WIFI;
        }
        return net;
    }

    public static int getMobileConnectType(Context context) {
        TelephonyManager tm = (TelephonyManager) context
                .getSystemService(Context.TELEPHONY_SERVICE);
        int subType = tm.getNetworkType();
        switch (subType) {
            case 13: //TelephonyManager.NETWORK_TYPE_LTE
                return CONNECT_4G;
            case TelephonyManager.NETWORK_TYPE_HSDPA:
            case TelephonyManager.NETWORK_TYPE_HSPA:
            case TelephonyManager.NETWORK_TYPE_HSUPA:
            case TelephonyManager.NETWORK_TYPE_EVDO_0:
            case TelephonyManager.NETWORK_TYPE_EVDO_A:
            case TelephonyManager.NETWORK_TYPE_UMTS:
            case 12: // TelephonyManager.NETWORK_TYPE_EVDO_B
            case 14: // TelephonyManager.NETWORK_TYPE_EHRPD
            case 15: // TelephonyManager.NETWORK_TYPE_HSPAP
                return CONNECT_3G;

            case TelephonyManager.NETWORK_TYPE_1xRTT:
            case TelephonyManager.NETWORK_TYPE_CDMA:
            case TelephonyManager.NETWORK_TYPE_EDGE:
            case TelephonyManager.NETWORK_TYPE_GPRS:
            case TelephonyManager.NETWORK_TYPE_IDEN:
                return CONNECT_2G;
            case TelephonyManager.NETWORK_TYPE_UNKNOWN:
            default:
                return CONNECT_MOBILE;
        }
    }

    public static HashMap<String, String> getNetworkInfo(Context context) {
        HashMap<String, String> map = new HashMap<>();
        ConnectivityManager cm = (ConnectivityManager) context.getSystemService(Context.CONNECTIVITY_SERVICE);
        if (cm != null) {
            NetworkInfo info = cm.getActiveNetworkInfo();
            if (info != null) {
                String typeName = info.getTypeName();
                String subTypeName = info.getSubtypeName();
                map.put("net_env", typeName);
                map.put("net_subtype", subTypeName);
                if (info.getType() == ConnectivityManager.TYPE_MOBILE) {
                    LogUtils.d("network", "typeName = " + typeName + ", subTypeName = " + subTypeName);
                    /**
                     TelephonyManager telephonyManager = (TelephonyManager)context.getSystemService(Context.TELEPHONY_SERVICE);
                     // for example value of first element
                     CellInfoGsm cellinfogsm = (CellInfoGsm)telephonyManager.getAllCellInfo().get(0);
                     CellSignalStrengthGsm cellSignalStrengthGsm = cellinfogsm.getCellSignalStrength();
                     cellSignalStrengthGsm.getDbm();
                     */
                } else if (info.getType() == ConnectivityManager.TYPE_WIFI) {
                    WifiManager wifiManager = (WifiManager) context.getSystemService(Context.WIFI_SERVICE);
                    WifiInfo wifiInfo = wifiManager.getConnectionInfo();
                    if (wifiInfo != null) {
                        int rssi = wifiInfo.getRssi();
                        String ssid = wifiInfo.getSSID();
                        map.put("net_rssi", Integer.toString(rssi));
                        map.put("net_ssid", ssid);
                        LogUtils.d("network", "typeName = " + typeName + ", subTypeName = " + subTypeName + ", rssi = " + rssi + ", ssid = " + ssid);
                    }
                }
            }else{
                map.put("activeNetwork", "null");
            }
        } else {
            map.put("networkInfo", "null");
        }
        return map;
    }

    /**
     * definition:
     * 0 - NONE
     * 1 - WIFI
     * 2 - 2G
     * 3 - 3G
     * 4 - 4G
     * 5 - Mob
     * @param context
     * @return
     */
    public static int getNetworkType(Context context) {
        boolean connected = NetworkUtils.isNetworkAvailable(context);
        if (!connected) {
            return 0;
        }
        int strType = 0;
        int type = getConnectType(context);
        switch (type) {
            case NetworkUtils.CONNECT_2G:
                strType = 2;
                break;
            case NetworkUtils.CONNECT_3G:
                strType = 3;
                break;
            case NetworkUtils.CONNECT_4G:
                strType = 4;
                break;
            case NetworkUtils.CONNECT_MOBILE:
                strType = 5;
                break;
            case NetworkUtils.CONNECT_WIFI:
                strType = 1;
                break;
            case NetworkUtils.CONNECT_NONE:
                strType = 0;
                break;
        }
        return strType;
    }

}
