package ai.fedml.edge.utils;

import android.content.Context;
import android.os.Environment;
import android.os.StatFs;

import java.io.BufferedInputStream;
import java.io.BufferedOutputStream;
import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;

import ai.fedml.edge.service.ContextHolder;
import androidx.annotation.NonNull;

public class StorageUtils {
    private static final String TAG = "StorageUtils";
    private static final String LOG_DIR = "logs";
    private static final String MODEL_DIR = "models";
    private static final String DATASET_DIR = "dataset";
    private static final int BUFFER_SIZE = 8192;

    private StorageUtils() {
    }

    public static String getSdCardPath() {
        return Environment.getExternalStorageDirectory().getAbsolutePath();
    }

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

    public static String getDatasetPath() {
        Context context = ContextHolder.getAppContext();
        File dirFile = context.getExternalFilesDir(DATASET_DIR);
        if (dirFile != null) {
            return dirFile.getAbsolutePath();
        }
        dirFile = context.getDir(DATASET_DIR, Context.MODE_PRIVATE);
        if (dirFile == null) {
            throw new RuntimeException("Model Path create failed!");
        }
        return dirFile.getAbsolutePath();
    }

    /**
     * save data to Label Data Directory
     *
     * @param data     data
     * @param fileName FileName,relative the Label Data Path
     * @return filePath if success return the file path, or null.
     */
    public static String saveToDatasetPath(@NonNull InputStream data, String fileName) {
        BufferedOutputStream bos = null;
        final String dataDir = getDatasetPath();
        final String filePath = dataDir + File.separator + fileName;
        LogHelper.d("StorageUtils saveToLabelDataPath: %s", filePath);
        try {
            writeToFile(data, filePath);
            return filePath;
        } catch (IOException e) {
            LogHelper.e(e, "StorageUtils saveToLabelDataPath failed");
        }
        return null;
    }

    private static void writeToFile(@NonNull InputStream data, @NonNull String destFilePath) throws IOException {
        File destFile = new File(destFilePath);
        createParentDirs(destFile);
        try (BufferedInputStream bis = new BufferedInputStream(data);
             BufferedOutputStream bos = new BufferedOutputStream(new FileOutputStream(destFilePath))) {
            int bytesRead = 0;
            byte[] buffer = new byte[BUFFER_SIZE];
            while ((bytesRead = bis.read(buffer, 0, BUFFER_SIZE)) != -1) {
                bos.write(buffer, 0, bytesRead);
            }
            bos.flush();
        } catch (IOException e) {
            LogHelper.e(e, "StorageUtils writeToFile failed");
            throw e;
        }
    }

    private static void createParentDirs(File file) throws IOException {
        if (file == null) {
            return;
        }
        File parent = file.getCanonicalFile().getParentFile();
        if (parent == null) {
            return;
        }
        parent.mkdirs();
        if (!parent.isDirectory()) {
            throw new IOException("Unable to create parent directories of " + file);
        }
    }
}
