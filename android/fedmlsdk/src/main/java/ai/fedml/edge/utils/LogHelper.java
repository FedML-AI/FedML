package ai.fedml.edge.utils;

import android.util.Log;

import com.google.common.collect.ImmutableMap;

import java.util.LinkedList;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.concurrent.locks.Lock;
import java.util.concurrent.locks.ReentrantReadWriteLock;

import ai.fedml.edge.BuildConfig;
import ai.fedml.edge.utils.preference.SharePreferencesData;

public class LogHelper {
    private static final String TAG = "FedML-Mobile-Client";
    private static final int LOG_LINE_NUMBER = 100;

    private static final boolean DEBUG = true;
    private static final Map<Integer, String> LEVEL_MAP = ImmutableMap.of(Log.VERBOSE, "VERBOSE", Log.DEBUG, "DEBUG",
            Log.INFO, "INFO", Log.WARN, "WARN", Log.ERROR, "ERROR");
    private static final ReentrantReadWriteLock rwl = new ReentrantReadWriteLock();
    private static final Lock rLock = rwl.readLock();
    private static final Lock wLock = rwl.writeLock();
    private static List<String> sLogLines = new LinkedList<>();

    public static void v(Object arg) {
        if (DEBUG) {
            String message = arg == null ? "null" : arg.toString();
            print(Log.VERBOSE, TAG, buildMessage("%s", message));
        }
    }

    public static void v(String format, Object... args) {
        if (DEBUG) {
            print(Log.VERBOSE, TAG, buildMessage(format, args));
        }
    }

    public static void d(Object arg) {
        if (DEBUG) {
            String message = arg == null ? "null" : arg.toString();
            print(Log.DEBUG, TAG, buildMessage("%s", message));
        }
    }

    public static void d(String format, Object... args) {
        if (DEBUG) {
            print(Log.DEBUG, TAG, buildMessage(format, args));
        }
    }

    public static void e(Object arg) {
        String message = arg == null ? "null" : arg.toString();
        print(Log.ERROR, TAG, buildMessage("%s", message));
    }

    public static void e(String format, Object... args) {
        print(Log.ERROR, TAG, buildMessage(format, args));
    }

    public static void i(String format, Object... args) {
        print(Log.INFO, TAG, buildMessage(format, args));
    }

    public static void e(Throwable tr, String format, Object... args) {
        print(Log.ERROR, TAG, buildMessage(format, args) + '\n' + Log.getStackTraceString(tr));
    }

    public static void wtf(String format, Object... args) {
        print(Log.WARN, TAG, buildMessage(format, args));
    }

    public static void wtf(Throwable tr, String format, Object... args) {
        print(Log.WARN, TAG, buildMessage(format, args) + '\n' + Log.getStackTraceString(tr));
    }

    /**
     * Formats the caller's provided message and prepends useful info like
     * calling thread ID and method name.
     */
    private static String buildMessage(String format, Object... args) {
        String msg = (args == null) ? format : String.format(Locale.US, format, args);
        StackTraceElement[] trace = new Throwable().fillInStackTrace().getStackTrace();
        String caller = "<unknown>";
        for (int i = 2; i < trace.length; i++) {
            Class<?> clazz = trace[i].getClass();
            if (!LogHelper.class.equals(clazz) && !trace[i].isNativeMethod() && !Thread.class.equals(clazz)) {
                caller = "(" + trace[i].getFileName() + ":" + trace[i].getLineNumber() + ")";
                break;
            }
        }
        return String.format(Locale.US, "[@device-id-%s][thread-%d]%s: %s", SharePreferencesData.getBindingId(),
                Thread.currentThread().getId(), caller, msg);
    }

    private static void print(int priority, String tag, String msg) {
        String log = "[" + getLevel(priority) + "] " + tag + " " + msg;
        wLock.lock();
        try {
            sLogLines.add(log);
        } finally {
            wLock.unlock();
        }
        Log.println(priority, tag, msg);
    }

    private static String getLevel(int priority) {
        return LEVEL_MAP.get(priority);
    }

    public static List<String> getLogLines() {
        rLock.lock();
        try {
            if (sLogLines.size() > 0) {
                List<String> logs = sLogLines;
                sLogLines = new LinkedList<>();
                return logs;
            }
            return null;
        } finally {
            rLock.unlock();
        }
    }
}
