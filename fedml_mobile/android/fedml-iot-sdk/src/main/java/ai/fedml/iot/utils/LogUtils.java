package ai.fedml.iot.utils;

import android.os.Environment;
import android.util.Log;

import ai.fedml.iot.ApplicationContextHolder;
import ai.fedml.iot.Config;
import ai.fedml.iot.device.IoTDevice;

import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.text.SimpleDateFormat;
import java.util.Date;
import java.util.Locale;

public class LogUtils {
    private static String TAG = "_IOTClient_";
    private static boolean sOpenLog = false;
    private final static boolean sOpenLogToFile = false;
    private static FileWriter mFileWriter = null;

    public static void setLogEnable(boolean enable){
        TAG = TAG + IoTDevice.getAppPackageName(ApplicationContextHolder.getContext()) + "."
                + IoTDevice.getAppVersionName(ApplicationContextHolder.getContext());
        sOpenLog = enable;
    }

    public static boolean isLogEnable(){
        return sOpenLog;
    }

    public static void d(String logKey, String msg) {
        if (sOpenLog) {
            StackTraceElement ste = new Throwable().getStackTrace()[1];
            StringBuilder sb = new StringBuilder();
            sb.append("[").append(logKey).append("]").append(build(msg,ste));
            String log = sb.toString();
            Log.d(TAG, log);
            if (sOpenLogToFile) {
                writeToFile(log);
            }
        }
    }

    public static void fd(String logKey, String msg){
        StackTraceElement ste = new Throwable().getStackTrace()[1];
        StringBuilder sb = new StringBuilder();
        sb.append("[").append(logKey).append("]").append(build(msg,ste));
        String log = sb.toString();
        Log.d(TAG, log);
        if (sOpenLogToFile) {
            writeToFile(log);
        }
    }

    public static void d(String msg) {
        if (sOpenLog) {
            StackTraceElement ste = new Throwable().getStackTrace()[1];
            String log = build(msg, ste);
            Log.d(TAG, log);
        }
    }

    public static void i(String logKey, String msg) {
        if (sOpenLog) {
            StackTraceElement ste = new Throwable().getStackTrace()[1];
            String log = build(msg, ste);
            Log.i(TAG, "[" + logKey + "]" + log);
        }
    }

    public static void fi(String logKey, String msg) {
        StackTraceElement ste = new Throwable().getStackTrace()[1];
        String log = build(msg, ste);
        Log.i(TAG, "[" + logKey + "]" + log);
    }

    public static void v(String logKey, String msg) {
        if (sOpenLog) {
            StackTraceElement ste = new Throwable().getStackTrace()[1];
            String log = build(msg, ste);
            Log.v(TAG, "[" + logKey + "]" + log);
        }
    }

    public static void w(String logKey, String msg) {
        if (sOpenLog) {
            StackTraceElement ste = new Throwable().getStackTrace()[1];
            String log = build(logKey, msg, ste);
            Log.w(TAG, log);
            if (sOpenLogToFile) {
                writeToFile(log);
            }
        }
    }

    public static void e(String logKey, String msg) {
        if (sOpenLog) {
            StackTraceElement ste = new Throwable().getStackTrace()[1];
            String log = build(logKey, msg, ste);
            Log.e(TAG, log);
            if (sOpenLogToFile) {
                writeToFile(log);
            }
        }
    }

    public static void fe(String logKey, String msg) {
        StackTraceElement ste = new Throwable().getStackTrace()[1];
        String log = build(logKey, msg, ste);
        Log.e(TAG, log);
        if (sOpenLogToFile) {
            writeToFile(log);
        }
    }

    public static void e(String logKey, String msg, Throwable e) {
        if (sOpenLog) {
            StackTraceElement ste = new Throwable().getStackTrace()[1];
            String log = build(logKey, msg, ste);
            Log.e(TAG, log);
            if (sOpenLogToFile) {
                writeToFile(log);
            }
        }
    }

    public static void fe(String logKey, String msg, Throwable e) {
        StackTraceElement ste = new Throwable().getStackTrace()[1];
        String log = build(logKey, msg, ste);
        Log.e(TAG, log, e);
        if (sOpenLogToFile) {
            writeToFile(build(logKey, msg, ste, e));
        }
    }

    private static void writeToFile(String strLog) {
        if (!sOpenLogToFile) {
            return;
        }
        if (mFileWriter == null) {
            SimpleDateFormat dateFormat = new SimpleDateFormat("yyyy-MM-dd", Locale.getDefault());
            String date = dateFormat.format(new Date());
            String logFilePath = Environment.getExternalStorageDirectory().toString() + "/iovcoreBaseLog-" + date + ".txt";
            try {
                mFileWriter = new FileWriter(logFilePath, true);
            } catch (IOException e) {
                Log.e(TAG, "create file writer fail", e);
            }
        }

        try {
            mFileWriter.write(new SimpleDateFormat("yyyy-MM-dd HH:mm:ss", Locale.getDefault()).format(new Date(System.currentTimeMillis())));
            mFileWriter.write(":");
            mFileWriter.write(strLog);
            mFileWriter.write("\r\n");
            mFileWriter.flush();
        } catch (IOException e) {
            Log.e(TAG, "write file fail", e);
        }
    }

    private static void logToFile(String tag, String strLog, String fileName) {
        Log.d(tag, strLog);
        String dirPath = Environment.getExternalStorageDirectory().getAbsolutePath() + "/android/iovcore/log";
        File dir = new File(dirPath);
        if (!dir.exists()){
            dir.mkdirs();
        }
        String logFilePath = dirPath + "/" + fileName;
        FileWriter fileWriter = null;
        try {
            fileWriter = new FileWriter(logFilePath, true);
            fileWriter.write(new SimpleDateFormat("yyyy-MM-dd HH:mm:ss", Locale.getDefault()).format(new Date(System.currentTimeMillis())));
            fileWriter.write(": ");
            fileWriter.write(strLog);
            fileWriter.write("\r\n");
            fileWriter.flush();
            fileWriter.close();
        } catch (IOException e) {
            Log.e(tag, "create file writer fail", e);
        }
    }

    /** 是否记录账号相关日志到本地文件 */
    public static boolean LogAccountEnable = false;
    public static void logAccount(String log){
        if (LogAccountEnable) {
            logToFile("account-log", log, "logaccount");
        }
    }

    public static boolean mSaveCloudLog = false;
    public static void logConnection(String msg){
        StackTraceElement ste = new Throwable().getStackTrace()[1];
        String log = "(" + Thread.currentThread() + "): " + build(msg, ste);
        if (sOpenLog) {
            Log.d("connection", log);
        }
        if (mSaveCloudLog){
            SimpleDateFormat dateFormat = new SimpleDateFormat("yyyy-MM-dd HH:mm:ss.SSS", Locale.getDefault());
            String date = dateFormat.format(new Date());

            String dirPath = Environment.getExternalStorageDirectory().getAbsolutePath() + Config.SDCARD_PATH + "log";
            File dir = new File(dirPath);
            if (!dir.exists()) {
                dir.mkdirs();
            }
            String logFilePath = dirPath + "/connect.log";
            FileWriter fileWriter = null;
            try {
                fileWriter = new FileWriter(logFilePath, true);
                fileWriter.write(date);
                fileWriter.write(":");
                fileWriter.write(log);
                fileWriter.write("\r\n");
                fileWriter.flush();
                fileWriter.close();
            } catch (IOException e) {
                Log.e(TAG, "create file writer fail", e);
            }
        }
    }

    /**
     * 制作打log位置的文件名与文件行号详细信息
     *
     * @param log
     * @param ste
     * @return
     */
    private static String build(String log, StackTraceElement ste) {
        StringBuilder buf = new StringBuilder();
        if (ste.isNativeMethod()) {
            buf.append("(Native Method)");
        } else {
            String fName = ste.getFileName();

            if (fName == null) {
                buf.append("(Unknown Source)");
            } else {
                int lineNum = ste.getLineNumber();
                buf.append('(');
                buf.append(fName);
                if (lineNum >= 0) {
                    buf.append(':');
                    buf.append(lineNum);
                }
                buf.append("):");
            }
        }
        buf.append(log);
        return buf.toString();
    }

    private static String build(String logKey, String msg, StackTraceElement ste) {
        StringBuilder sb = new StringBuilder();
        sb.append("[").append(logKey).append("]").append(build(msg,ste));
        return sb.toString();
    }

    private static String build(String logKey, String msg, StackTraceElement ste, Throwable e) {
        StringBuilder sb = new StringBuilder();
        sb.append("[").append(logKey).append("]").append(ste.toString()).append(":").append(msg).append("\r\n").append("e:").append(e.getMessage());
        return sb.toString();
    }
}

