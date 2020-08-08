package ai.fedml.iot;


public class Config {
    public static final String COMMON_TAG = "iovplugin_";
    public static final boolean bDebugMode = false;

    public final static String BROADCAST_RELOAD_PLUGIN = "ai.fedml.iovclient.RELOAD_PLUGIN";

    public final static String KEY_SP_LATEST_USED_PLUGIN_VERSION = "KEY_SP_LATEST_USED_PLUGIN_VERSION";

    public static final int DEVICE_TYPE = 2;

    public static final int OS_TYPE = 1;

    public final static String SDCARD_PATH = "/iotsupercloud/iovcore/";
    public final static String SDCARD_CACHE_PATH = "/cache/";
    public final static String SDCARD_TEMP_PATH = "/temp/";
    public final static String SDCARD_LOG_PATH = "/log/";
    public final static String SDCARD_LOCATION_TRACK_PATH = "gpstrack/";

    public static String BASE_URL_IOT_APP_SERVICE = "http://app.iotsupercloud.com";
    //public static String BASE_URL_IOT_APP_SERVICE = "http://193.112.1.113:3001";

    public static String BASE_URL_IOT_MQ_SERVICE = "tcp://mq.iotsupercloud.com:1883";
    //public static String BASE_URL_IOT_MQ_SERVICE = "tcp://111.230.252.232:1883";//A
    //public static String BASE_URL_IOT_MQ_SERVICE = "tcp://118.89.64.130:1883";//B
    //public static String BASE_URL_IOT_MQ_SERVICE = "tcp://193.112.12.67:1883";//C
    //public final static String BASE_URL_IOT_MQ_SERVICE = "tcp://118.89.61.144:1883";//TEST

    public static String MQ_USER = "admin";
    public static String MQ_PASSWORD = "admin";

    //FOUR hours, data size = (109 * 3600 * 4 / 1024) = 1532 Kb
    public static final int MAX_LOCATION_CACHE_NUMBER = 3600 * 4;

    /**
     * 配置文件中的版本好信息和这里的配置是有关系的,发布的时候注意
     * deviceearn/gradle.properties
     *
     * ai.fedml.pluginframework.Config#RELEASE == false
     * #version=*.*.*-TEST-SNAPSHOT
     * <p>
     * ai.fedml.pluginframework.Config#RELEASE == true
     * #version=*.*.*-SNAPSHOT
     */
    public static final boolean RELEASE = true;

    public static long CheckUpgradeTriggerInterval = RELEASE ? 15 * 60 * 1000 : 10 * 1000;
    public static long CheckUpgradeInterval = RELEASE ? 8 * 60 * 60 * 1000 : 2 * 10 * 1000;

    public static boolean CheckUpgradeAuto = true;
    public static String Signature = "308202cb308201b3a0030201020204191c5a2b300d06092a864886f70d0" +
            "1010b05003015311330110603550403130a74656e63656e744d61703020170d313531323032303631343" +
            "2325a180f33303135303430343036313432325a3015311330110603550403130a74656e63656e744d617" +
            "030820122300d06092a864886f70d01010105000382010f003082010a0282010100a216ea667f028523f" +
            "381a96bb71a5c7f245dd4d42c576226c17e8c72528a193161d264911f2473941f7b369eb8585c3f8a737" +
            "876eec52785019cdd189c38f65c9b8e83c750e38511e4068851d251e15963cc69af720c51b72b69e26e2" +
            "765a491a7a56f0eef27f45d9d8dc0add4144a4e583d73c38f0aa770768729b1af5df020b7ab71c731f97" +
            "47200cf4c353be4e8e13f7c77ca19ba9afa71493abf5fc59996a1c10552b103ed3bf59d5ce812e9cefc7" +
            "5511d7de1b64d8cad40ccbfc9321838874caec76d88ebb2802349daf4aeabfa833ba1ff14a2f19bc5f80" +
            "0df4cc98da8192c8ca9c04358791264eaa6191b06b63859682a4f5b43aa6f410fa74e723943020301000" +
            "1a321301f301d0603551d0e04160414708184523790894bfab54cf4a6595b3eec2a1f73300d06092a864" +
            "886f70d01010b050003820101007bb87842530b419e304549f22477ed8823d390493d3d08587f942a6f3" +
            "39926a1fe7354242036c53c13f8cac1b17b4f1f93ef2a7ae4631a0f7cb54f1157fa68e39aacb11ca934b" +
            "44f587b9ba32f0eac2a453881cf54a004ca3a666321859c00bcdc401f4f385b55bf6a096443f28d3a44a" +
            "b69a8d2fbba8b93466a400c0be3a9ae0d1eb37a57c69e0814834a959f5ac8ffda35ef17f7e131a4e4dfc" +
            "5052d6609d8ffa793627027f914b054800abb8e9b07cc1f5da6d3a380528fd6d27b1da414a721ff105fc" +
            "965a63898fed8a830b4c1d11bc1733f874526d7e731853da73d52010b3cf279022b0f7a486e53997ba96" +
            "a34a8a9fabbdd40ca27e49c9041c45c337e";
    public static int SignatureHash = 1218518767;

    //common attr
    public static String DeviceId = "uninitialized";
    public static String Channel = "uninitialized";

    //app attr
    public static int AndroidApi = 0;
    public static String AppPackageName = "uninitialized";
    public static String AppVersionName = "uninitialized";
    public static int AppVersionCode = -1;

    //plugin attr
    public static String PluginPackageName = "uninitialized";
    public static String PluginVersionName = "uninitialized";
    public static int PluginVersionCode = -1;

}
