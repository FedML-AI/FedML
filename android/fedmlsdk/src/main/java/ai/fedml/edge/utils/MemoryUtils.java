package ai.fedml.edge.utils;

import android.app.ActivityManager;
import android.content.Context;
import android.os.Environment;
import android.os.StatFs;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.text.DecimalFormat;

import ai.fedml.edge.utils.entity.Memory;

public class MemoryUtils {

    public static Memory getMemory(Context context) {
        Memory memory = new Memory();
        try {
            memory.setRamMemoryTotal(getTotalMemory());
            memory.setRamMemoryAvailable(getMemoryAvailable(context));
            memory.setRomMemoryAvailable(getRomSpaceAvailable());
            memory.setRomMemoryTotal(getRomSpaceTotal());
        } catch (Exception e) {
            LogHelper.i(e.toString());
        }
        return memory;
    }

    private static String getTotalMemory() {
        String str1 = "/proc/meminfo";
        String str2;
        String[] arrayOfString;
        long initial_memory = 0;
        try {
            FileReader localFileReader;

            localFileReader = new FileReader(str1);

            BufferedReader localBufferedReader = new BufferedReader(localFileReader, 8192);
            str2 = localBufferedReader.readLine();

            arrayOfString = str2.split("\\s+");

            initial_memory = Long.valueOf(arrayOfString[1]) * 1024;
            localBufferedReader.close();

        } catch (Exception e) {
            LogHelper.i(e.toString());
        }
        return convertBytesToGB(initial_memory);
    }

    private static String getMemoryAvailable(Context context) {
        ActivityManager am = (ActivityManager) context.getSystemService(Context.ACTIVITY_SERVICE);
        ActivityManager.MemoryInfo mi = new ActivityManager.MemoryInfo();
        if (am != null) {
            am.getMemoryInfo(mi);
        }
        return convertBytesToGB(mi.availMem);
    }

    private static String getRomSpaceAvailable() {
        File path = Environment.getDataDirectory();
        StatFs stat = new StatFs(path.getPath());
        long blockSize = stat.getBlockSize();
        long availableBlocks = stat.getAvailableBlocks();
        return convertBytesToGB(availableBlocks * blockSize);
    }

    private static String getRomSpaceTotal() {
        File path = Environment.getDataDirectory();
        StatFs stat = new StatFs(path.getPath());
        long blockSize = stat.getBlockSize();
        long totalBlocks = stat.getBlockCount();
        return convertBytesToGB(totalBlocks * blockSize);
    }

    public static String convertBytesToGB(long bytes) {
        // stay the same as Formatter.formatFileSize
        double gigabytes = bytes / (1000.0 * 1000.0 * 1000.0);
        DecimalFormat df = new DecimalFormat("#.##");
        return df.format(gigabytes);
    }

}
