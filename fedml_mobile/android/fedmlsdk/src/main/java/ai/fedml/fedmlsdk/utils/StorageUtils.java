package ai.fedml.fedmlsdk.utils;

import android.content.Context;
import android.os.StatFs;
import android.util.Log;

import androidx.annotation.NonNull;

import com.google.common.io.Files;

import java.io.BufferedInputStream;
import java.io.BufferedOutputStream;
import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;

import ai.fedml.fedmlsdk.ContextHolder;
import lombok.Cleanup;

public class StorageUtils {
    private static final String LOG_DIR = "logs";
    private static final String MODEL_DIR = "models";
    private static final String LABEL_DATA_DIR = "labeldata";
    public static final int BUFFER_SIZE = 8192;

    /**
     * get dir Available size
     *
     * @return Bytes
     */
    public static long getAvailableSize(@NonNull final String dir) {
        StatFs sf = new StatFs(dir);
        return sf.getAvailableBytes();
    }

    public static String getLogPath() {
        Context context = ContextHolder.getAppContext();
        File dirFile = context.getExternalFilesDir(LOG_DIR);
        if (dirFile != null) {
            return dirFile.getAbsolutePath();
        }
        dirFile = context.getDir(LOG_DIR, Context.MODE_PRIVATE);
        if (dirFile == null) {
            throw new RuntimeException("Log Path create failed!");
        }
        return dirFile.getAbsolutePath();
    }

    public static String getModelPath() {
        Context context = ContextHolder.getAppContext();
        File dirFile = context.getExternalFilesDir(MODEL_DIR);
        if (dirFile != null) {
            return dirFile.getAbsolutePath();
        }
        dirFile = context.getDir(MODEL_DIR, Context.MODE_PRIVATE);
        if (dirFile == null) {
            throw new RuntimeException("Model Path create failed!");
        }
        return dirFile.getAbsolutePath();
    }

    public static String getLabelDataPath() {
        Context context = ContextHolder.getAppContext();
        File dirFile = context.getExternalFilesDir(LABEL_DATA_DIR);
        if (dirFile != null) {
            return dirFile.getAbsolutePath();
        }
        dirFile = context.getDir(LABEL_DATA_DIR, Context.MODE_PRIVATE);
        if (dirFile == null) {
            throw new RuntimeException("Model Path create failed!");
        }
        return dirFile.getAbsolutePath();
    }

    /**
     * save data to Directory
     *
     * @param data     data
     * @param fileName FileName,relative the Label Data Path
     * @return
     */
    public static boolean saveToLabelDataPath(byte[] data, String fileName) {
        BufferedOutputStream bos = null;
        final String dataDir = getLabelDataPath();
        final String filePath = Files.simplifyPath(dataDir + File.separator + fileName);
        File destFile = new File(filePath);
        try {
            Files.createParentDirs(destFile);
            Files.write(data, destFile);
            return true;
        } catch (IOException e) {
            Log.e("StorageUtils", "saveToLabelDataPath", e);
        }
        return false;
    }

    /**
     * save data to Label Data Directory
     *
     * @param data     data
     * @param fileName FileName,relative the Label Data Path
     * @return save success
     */
    public static boolean saveToLabelDataPath(@NonNull InputStream data, String fileName) {
        BufferedOutputStream bos = null;
        final String dataDir = getLabelDataPath();
        final String filePath = Files.simplifyPath(dataDir + File.separator + fileName);
        Log.d("StorageUtils", "saveToLabelDataPath: " + filePath);
        try {
            writeToFile(data, filePath);
            return true;
        } catch (IOException e) {
            Log.e("StorageUtils", "saveToLabelDataPath", e);
        }
        return false;
    }

    /**
     * save data to Model Directory
     *
     * @param data     data
     * @param fileName FileName,relative the Label Data Path
     * @return
     */
    public static String saveToModelPath(byte[] data, String fileName) {
        BufferedOutputStream bos = null;
        final String dataDir = getModelPath();
        final String filePath = Files.simplifyPath(dataDir + File.separator + fileName);
        File destFile = new File(filePath);
        try {
            Files.createParentDirs(destFile);
            Files.write(data, destFile);
            return filePath;
        } catch (IOException e) {
            Log.e("StorageUtils", "saveToLabelDataPath", e);
        }
        return null;
    }

    private static void writeToFile(@NonNull InputStream data, @NonNull String destFilePath) throws IOException {
        File destFile = new File(destFilePath);
        Files.createParentDirs(destFile);
        @Cleanup
        BufferedInputStream bis = new BufferedInputStream(data);
        @Cleanup
        BufferedOutputStream bos = new BufferedOutputStream(new FileOutputStream(destFile));
        int bytesRead = 0;
        byte[] buffer = new byte[BUFFER_SIZE];
        while ((bytesRead = bis.read(buffer, 0, 8192)) != -1) {
            bos.write(buffer, 0, bytesRead);
        }
        bos.flush();
    }
}
