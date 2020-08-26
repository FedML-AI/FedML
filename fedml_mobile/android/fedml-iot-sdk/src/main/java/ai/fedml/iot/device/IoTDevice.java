package ai.fedml.iot.device;

import android.content.Context;

import ai.fedml.iot.CommonConfig;
import ai.fedml.iot.utils.LogUtils;

public class IoTDevice {
    private static final String TAG = IoTDevice.class.getClass().getSimpleName();

    private static String IoTDeviceID = "";
    private static String ChannelID = CommonConfig.CHANNEL_ID;
    //private static int DeviceType = Config.DEVICE_TYPE;
    //private static int OSType = Config.OS_TYPE;
    private static String HardwareDeviceID = ""; //defined by hardware manufacturer, can be unique among this channel.
    //private static String WifiMac = "";
    //private static String IMEI = "";
    //private static String OSVersionName = "";
    private static String AppPackageName = "";
    private static String AppVersionName = "";
    private static int AppVersionCode = 0;

    //private static String DeviceBrand = "";
    //private static String Manufacturer = "";
    //private static String Model = "";
    //private static String Product = "";
    //private static String Hardware = "";
    //private static String Board = "";
    //private static String Device = "";

    //private static String strIOTDeviceInfo = "";

    private static boolean bInit = false;
    public static void init(Context context){
        HardwareDeviceID = DeviceUtil.getCID();
        LogUtils.d(TAG, "DeviceID = " + HardwareDeviceID);

        AppPackageName = CommonConfig.AppPackageName;
        LogUtils.d(TAG, "AppPackageName = " + AppPackageName);

        AppVersionName = CommonConfig.AppVersionName;
        LogUtils.d(TAG, "AppVersionName = " + AppVersionName);

        AppVersionCode =CommonConfig.AppVersionCode;
        LogUtils.d(TAG, "AppVersionCode = " + AppVersionCode);

        StringBuilder sb = new StringBuilder();
        sb.append("domain-common");
        sb.append(ChannelID);
        sb.append(HardwareDeviceID);

        // TODO: generate Device_ID
        IoTDeviceID = "todo:generate_device_id";

        LogUtils.d(TAG, "FedMLDevice = " + IoTDeviceID);

        bInit = true;
    }

    public static String getDeviceID(Context context){
        if(!bInit) init(context);
        return IoTDeviceID;
    }

    public static String getDeviceID(){
        return IoTDeviceID;
    }

    public static String getChannelID(){
        return ChannelID;
    }

    public static String getAppVersionName(Context context){
        if(!bInit) init(context);
        return AppVersionName;
    }

    public static int getAppVersionCode(Context context){
        if(!bInit) init(context);
        return AppVersionCode;
    }

    public static String getAppPackageName(Context context){
        if(!bInit) init(context);
        return AppPackageName;
    }
}
