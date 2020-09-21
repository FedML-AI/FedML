package ai.fedml.iot;

public class CommonConfig {
    public static final String AppPackageName = "ai.fedml.iovclient";
    public static final String AppVersionName = "1.1.0";
    public static final int AppVersionCode = 220;

    public static final String CHANNEL_ID = "fedml.ai.10000";

    public static final int DEVICE_TYPE = 2;

    public static final int OS_TYPE = 1;

    public final static String SDCARD_PATH = "/iotsupercloud/iovcore/";
    public final static String SDCARD_CACHE_PATH = "/cache/";
    public final static String SDCARD_TEMP_PATH = "/temp/";
    public final static String SDCARD_LOG_PATH = "/log/";
    public final static String SDCARD_LOCATION_TRACK_PATH = "gpstrack/";

    //customized system broadcast
    public final static String BROADCAST_TEST = "ai.fedml.iovclient.TEST";
    public final static String BROADCAST_RUILIAN_ACC_OFF = "com.android.rmt.ACTION_ACC_OFF";
    public final static String BROADCAST_RUILIAN_ACC_ON = "com.android.rmt.ACTION_ACC_ON";

    public static final String INTENT_KEY_BUNDLE = "bundle";
    public static final String BUNDLE_KEY_COMMAND = "command";
}
