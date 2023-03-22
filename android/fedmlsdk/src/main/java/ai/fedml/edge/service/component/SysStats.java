package ai.fedml.edge.service.component;

import android.app.ActivityManager;
import android.content.Context;
import android.net.TrafficStats;
import android.os.Debug;
import android.os.Environment;
import android.os.StatFs;
import android.os.Process;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.IOException;
import java.io.RandomAccessFile;
import java.util.Arrays;

import ai.fedml.edge.service.ContextHolder;
import ai.fedml.edge.utils.CpuUtil;
import ai.fedml.edge.utils.LogHelper;
import lombok.Builder;
import lombok.Data;
import lombok.ToString;

public class SysStats {
    private static final int BUF_LEN = 8192;

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
        return CpuUtil.INSTANCE.getCpuUsage();
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


    private String[] getCpuInfo() {
        String[] arrayOfString;
        try (BufferedReader localBufferedReader = new BufferedReader(new FileReader("/proc/cpuinfo"), BUF_LEN)) {
            String str2 = localBufferedReader.readLine();
            arrayOfString = str2.split("\\s+");
            String[] cpuInfo = {"", ""};
            for (int i = 2; i < arrayOfString.length; i++) {
                cpuInfo[0] = cpuInfo[0] + arrayOfString[i] + " ";
            }
            str2 = localBufferedReader.readLine();
            arrayOfString = str2.split("\\s+");
            cpuInfo[1] += arrayOfString[2];
            return cpuInfo;
        } catch (IOException e) {
            LogHelper.e(e, "getCpuInfo failed.");
        }
        return null;
    }

    private float getCpuTotal() {
        String[] cpuInfo = getCpuInfo();
        if (cpuInfo != null) {
            try {
                return Float.parseFloat(cpuInfo[1]);
            } catch (NumberFormatException e) {
                LogHelper.e(e, "parseFloat CpuInfo failed." + Arrays.toString(cpuInfo));
            }
        }
        return 0.1f;
    }

    public int getCpuUsage() {
        try (RandomAccessFile reader = new RandomAccessFile("/proc/stat", "r")) {
            String load = reader.readLine();
            String[] toks = load.split(" ");
            long idle1 = Long.parseLong(toks[4]);
            long cpu1 = Long.parseLong(toks[2]) + Long.parseLong(toks[3]) + Long.parseLong(toks[4])
                    + Long.parseLong(toks[6]) + Long.parseLong(toks[7]) + Long.parseLong(toks[8]);

            try {
                Thread.sleep(360);
            } catch (Exception e) {
                e.printStackTrace();
            }

            reader.seek(0);
            load = reader.readLine();
            reader.close();
            toks = load.split(" ");
            long idle2 = Long.parseLong(toks[5]);
            long cpu2 = Long.parseLong(toks[2]) + Long.parseLong(toks[3]) + Long.parseLong(toks[4])
                    + Long.parseLong(toks[6]) + Long.parseLong(toks[7]) + Long.parseLong(toks[8]);
            return (int) (100 * (cpu2 - cpu1) / ((cpu2 + idle2) - (cpu1 + idle1)));

        } catch (IOException ex) {
            ex.printStackTrace();
        } catch (Exception e) {

        }
        return 0;
    }
}
