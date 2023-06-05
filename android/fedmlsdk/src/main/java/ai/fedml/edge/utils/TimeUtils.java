package ai.fedml.edge.utils;

public class TimeUtils {

    private static long serverSendTime = 0;

    private static long serverRecvTime = 0;

    private static long deviceSendTime = 0;

    private static boolean sync = false;

    public static void fillTime(long serverSendTime, long serverRecvTime, long deviceSendTime) {
        TimeUtils.serverSendTime = serverSendTime;
        TimeUtils.serverRecvTime = serverRecvTime;
        TimeUtils.deviceSendTime = deviceSendTime;
        sync = true;
    }

    public static long getAccurateTime() {
        if (sync) {
            return (serverRecvTime + serverSendTime + System.currentTimeMillis() - deviceSendTime) / 2;
        }
        return System.currentTimeMillis();
    }
}
