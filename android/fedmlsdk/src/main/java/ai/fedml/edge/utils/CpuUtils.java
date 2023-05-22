package ai.fedml.edge.utils;

import android.os.Build;
import android.text.TextUtils;
import android.util.Log;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileFilter;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.RandomAccessFile;
import java.util.regex.Pattern;

public class CpuUtils {

    private RandomAccessFile procStatFile;

    private RandomAccessFile appStatFile;

    private Long lastCpuTime;

    private Long lastAppCpuTime;
    
    private CpuUtils() {

    }

    private final static class LazyHolder {
        private final static CpuUtils sInstance = new CpuUtils();
    }

    public static CpuUtils getInstance() {
        return CpuUtils.LazyHolder.sInstance;
    }

    public Float getCpuUsage() {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
            return getCpuUsageForHigherVersion();
        } else {
            return getCpuUsageForLowerVersion();
        }
    }

    private Float getCpuUsageForLowerVersion() {
        Long cpuTime = 0L;
        Long appTime = 0L;
        Float value = 0.0f;
        try {
            if (procStatFile == null || appStatFile == null) {
                procStatFile = new RandomAccessFile("/proc/stat", "r");
                appStatFile = new RandomAccessFile("/proc/" + android.os.Process.myPid() + "/stat", "r");
            } else {
                procStatFile.seek(0L);
                appStatFile.seek(0L);
            }

            String procStatString = procStatFile.readLine();
            String appStatString = appStatFile.readLine();
            String[] procStats = procStatString.split(" ");
            String[] appStats = appStatString.split(" ");
            cpuTime = Long.parseLong(procStats[2]) + Long.parseLong(procStats[3])
                    + Long.parseLong(procStats[4]) + Long.parseLong(procStats[5])
                    + Long.parseLong(procStats[6]) + Long.parseLong(procStats[7])
                    + Long.parseLong(procStats[8]);
            appTime = Long.parseLong(appStats[13]) + Long.parseLong(appStats[14]);

            if (lastAppCpuTime == null && lastCpuTime == null) {
                lastAppCpuTime = appTime;
                lastCpuTime = cpuTime;
                return value;
            }
            value = (float) (appTime - lastAppCpuTime) / (float) (cpuTime - lastCpuTime) / (float) (appTime - lastAppCpuTime) * 100.0f;
            lastCpuTime = cpuTime;
            lastAppCpuTime = appTime;
        } catch (Exception e) {
            e.printStackTrace();
        }
        return value;
    }

    private Float getCpuUsageForHigherVersion() {
        LogHelper.i("getCpuUsageForHigherVersion");
        Process process = null;
        try {
            process = Runtime.getRuntime().exec("top -n 1");
            BufferedReader reader = new BufferedReader(new InputStreamReader(process.getInputStream()));
            String line;
            int cpuIndex = -1;
            while ((line = reader.readLine()) != null) {
                line = line.trim();
                if (TextUtils.isEmpty(line)) {
                    continue;
                }
                int tempIndex = getCPUIndex(line);
                if (tempIndex != -1) {
                    cpuIndex = tempIndex;
                    continue;
                }
                if (line.startsWith(String.valueOf(android.os.Process.myPid()))) {
                    if (cpuIndex == -1) {
                        continue;
                    }
                    String[] param = line.split("\\s+");
                    if (param.length <= cpuIndex) {
                        continue;
                    }
                    String cpu = param[cpuIndex];
                    if (cpu.endsWith("%")) {
                        cpu = cpu.substring(0, cpu.lastIndexOf("%"));
                    }
                    float rate = Float.parseFloat(cpu) / Runtime.getRuntime().availableProcessors();
                    return rate;
                }
            }
        } catch (IOException e) {
            throw new RuntimeException(e);
        } finally {
            if (null != process) {
                process.destroy();
            }
        }

        return null;
    }

    private int getCPUIndex(String line) {
//        LogHelper.i("getCPUIndex,%s", line);
        if (line.contains("CPU")) {
            String[] titles = line.split("\\s+");
            for (int i=0; i<titles.length; i++) {
//                LogHelper.i("getCPUIndex title, %s", titles[i]);
                if (titles[i].contains("CPU")) {
                    return i;
                }
            }
        }
        return -1;
    }

    public int getCores() {
        int cores;
        try {
            cores = new File("/sys/devices/system/cpu/").listFiles(CPU_FILTER).length;
        } catch (SecurityException e) {
            cores = 0;
        }
        return cores;
    }

    private final FileFilter CPU_FILTER = new FileFilter() {
        @Override
        public boolean accept(File pathname) {
            return Pattern.matches("cpu[0-9]", pathname.getName());
        }
    };

    public String getCpuAbi() {
        String cpuAbi;
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.LOLLIPOP) {
            cpuAbi = Build.SUPPORTED_ABIS[0];
        } else {
            cpuAbi = Build.CPU_ABI;
        }
        return cpuAbi;
    }

}































