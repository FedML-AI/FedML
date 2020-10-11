package ai.fedml.fedmlsdk.utils;

import com.tencent.mars.xlog.Log;
import com.tencent.mars.xlog.Xlog;

import java.text.SimpleDateFormat;
import java.util.Date;
import java.util.Locale;

import ai.fedml.fedmlsdk.BuildConfig;
import ai.fedml.fedmlsdk.ContextHolder;

public class LogUtils {
    static {
        System.loadLibrary("c++_shared");
        System.loadLibrary("marsxlog");
    }

    public static void init() {
        final String logPath = StorageUtils.getLogPath();
        // this is necessary, or may crash for SIGBUS
        final String cachePath = ContextHolder.getAppContext().getFilesDir() + "/xlog";
        //init xlog
        Xlog.XLogConfig logConfig = new Xlog.XLogConfig();
        logConfig.mode = Xlog.AppednerModeAsync;
        logConfig.logdir = logPath;
        SimpleDateFormat sdf = new SimpleDateFormat("yyyyMMdd", Locale.US);
        String timeStr = sdf.format(new Date());
        logConfig.nameprefix = "FedML" + timeStr;
        logConfig.pubkey = "";
        logConfig.compressmode = Xlog.ZLIB_MODE;
        logConfig.compresslevel = 0;
        logConfig.cachedir = "";
        logConfig.cachedays = 0;
        if (BuildConfig.DEBUG) {
            logConfig.level = Xlog.LEVEL_VERBOSE;
            Xlog.setConsoleLogOpen(true);
        } else {
            logConfig.level = Xlog.LEVEL_INFO;
            Xlog.setConsoleLogOpen(false);
        }
        Log.setLogImp(new Xlog());
    }

    public static void close() {
        Log.appenderClose();
    }
}
