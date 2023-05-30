package ai.fedml.edge.service.component;

import android.app.ActivityManager;
import android.content.Context;
import android.net.TrafficStats;
import android.os.Debug;
import android.os.Environment;
import android.os.StatFs;
import android.os.Process;

import java.io.File;

import ai.fedml.edge.service.ContextHolder;
import ai.fedml.edge.utils.CpuUtils;
import lombok.Builder;
import lombok.Data;
import lombok.ToString;

public class SysStats {

    private static class LazyHolder {
        private final static SysStats SYS_STATS = new SysStats();
    }

    @Data
    @Builder
    @ToString
    public static class MemoryStats {
        private float memoryUtilization;
        private float memoryInUse;
        private float memoryInUseSize;
        private float memoryAvailable;
    }

    public static SysStats getInstance() {
        return SysStats.LazyHolder.SYS_STATS;
    }

    private SysStats() {
    }

    public float getCpuUtilization() {
        return CpuUtils.getInstance().getCpuUsage();
    }

    public MemoryStats getMemoryInfo() {
        ActivityManager.MemoryInfo memoryInfo = new ActivityManager.MemoryInfo();
        ActivityManager activityManager =
                (ActivityManager) ContextHolder.getAppContext().getSystemService(Context.ACTIVITY_SERVICE);
        activityManager.getMemoryInfo(memoryInfo);
        float total = memoryInfo.totalMem / 1024f;
        float free = memoryInfo.availMem / 1024f;
        float used = total - free;
        float memoryUtilization = 100 * used / total;
        float memoryInUse = 0f;
        float memoryInUseSize = 0.0f;
        int[] pids = new int[]{Process.myPid()};
        Debug.MemoryInfo[] memoryInfoArray = activityManager.getProcessMemoryInfo(pids);
        if (memoryInfoArray != null && memoryInfoArray.length != 0) {
            Debug.MemoryInfo pidMemoryInfo = memoryInfoArray[0];
            // process memory In Used (MB)
            memoryInUse = pidMemoryInfo.getTotalPrivateDirty() / 1024f;
            memoryInUseSize = 100f * pidMemoryInfo.getTotalPrivateDirty() / memoryInfo.totalMem;
        }

        return MemoryStats.builder().memoryUtilization(memoryUtilization)
                .memoryInUse(memoryInUse).memoryInUseSize(memoryInUseSize).memoryAvailable(memoryInfo.availMem).build();
    }

    public int getProcessCpuThreadsInUse() {
        ThreadGroup group = Thread.currentThread().getThreadGroup();
        ThreadGroup topGroup = group;
        while (group != null) {
            topGroup = group;
            group = group.getParent();
        }

        if (topGroup == null) {
            return 0;
        }
        int slackSize = topGroup.activeCount() * 2;
        Thread[] slackThreads = new Thread[slackSize];
        return topGroup.enumerate(slackThreads);
    }

    public float getDiskUtilization() {
        File path = Environment.getExternalStorageDirectory();
        StatFs statFs = new StatFs(path.getPath());
        long blockSize = statFs.getBlockSizeLong();
        long availableBlocks = statFs.getAvailableBlocksLong();
        long totalSize = statFs.getBlockSizeLong() * statFs.getBlockCountLong();
        long availableSize = blockSize * availableBlocks;
        return 100F * availableSize / totalSize;
    }

    public long getNetworkTraffic() {
        return TrafficStats.getTotalRxBytes() + TrafficStats.getTotalTxBytes();
    }

}
