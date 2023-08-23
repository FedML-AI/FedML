package ai.fedml.edge.utils;

public class TimeUtils {

    private static long serverSendTime = 0L;

    private static long serverRecvTime = 0L;

    private static long deviceSendTime = 0L;

    private static long deviceRecvTime = 0L;

    private static boolean sync = false;

    private static long ntpTime = 0L;

    public static void fillTime(long serverSendTime, long serverRecvTime, long deviceSendTime, long deviceRecvTime) {
        TimeUtils.serverSendTime = serverSendTime;
        TimeUtils.serverRecvTime = serverRecvTime;
        TimeUtils.deviceSendTime = deviceSendTime;
        TimeUtils.deviceRecvTime = deviceRecvTime;
        TimeUtils.ntpTime = (serverRecvTime + serverSendTime + deviceRecvTime - deviceSendTime) / 2;
        sync = true;
    }

    public static long getAccurateTime() {
        if (sync) {
            return ntpTime + System.currentTimeMillis() - deviceRecvTime;
        }
        return System.currentTimeMillis();
    }
}
