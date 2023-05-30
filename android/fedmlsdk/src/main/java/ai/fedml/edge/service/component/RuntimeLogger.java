package ai.fedml.edge.service.component;

import android.os.Handler;

import java.util.ArrayList;
import java.util.List;

import ai.fedml.edge.request.RequestManager;
import ai.fedml.edge.request.listener.OnLogUploadListener;
import ai.fedml.edge.request.parameter.EdgesError;
import ai.fedml.edge.utils.BackgroundHandler;
import ai.fedml.edge.utils.LogHelper;

public class RuntimeLogger {
    private final Handler mBgHandler;
    private final long mEdgeId;
    private final long mRunId;
    private final Runnable mRunnable;

    public RuntimeLogger(final long edgeId, final long runId) {
        mEdgeId = edgeId;
        mRunId = runId;
        mBgHandler = new BackgroundHandler("LogUploader");
        LogHelper.resetLog();
        mRunnable = new Runnable() {
            @Override
            public void run() {
                flush();
                mBgHandler.postDelayed(this, 10000L);
            }
        };
    }

    public void start() {
        mBgHandler.postDelayed(mRunnable, 10000L);
    }

    public void release() {
        flush();
        mBgHandler.removeCallbacksAndMessages(null);
    }

    public void flush() {
        List<String> logs = LogHelper.getLogLines();
        uploadLog(logs);
    }

    private void uploadLog(final List<String> logs) {
        if (logs == null || logs.size() == 0) {
            return;
        }

        List<EdgesError> errorLines = new ArrayList<>();
        EdgesError error;
        for (int i = 0; i < logs.size(); ++i) {
            error = new EdgesError();
            String log = logs.get(i);
            if (log == null) {
                continue;
            }
            if (log.contains(" [ERROR] ")) {
                int errorLine = i + 1 + LogHelper.getLineNumber();
                error.setErrLine(errorLine);
                error.setErrMsg(log);
                errorLines.add(error);
            }
        }

        LogHelper.addLineNumber(logs.size());

        RequestManager.uploadLog(mRunId, mEdgeId, logs, errorLines, new OnLogUploadListener() {
            private int retryCnt = 3;

            @Override
            public void onLogUploaded(boolean success) {
                if (!success && retryCnt < 0) {
                    retryCnt--;
                    RequestManager.uploadLog(mRunId, mEdgeId, logs, errorLines,this);
                }
            }
        });
        MetricsReporter.getInstance().reportSystemMetric(mRunId, mEdgeId);
    }
}
