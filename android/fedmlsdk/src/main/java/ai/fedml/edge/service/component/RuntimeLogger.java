package ai.fedml.edge.service.component;

import android.os.Handler;
import android.util.Log;

import java.util.List;

import ai.fedml.edge.request.RequestManager;
import ai.fedml.edge.request.listener.OnLogUploadListener;
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
        mRunnable = new Runnable() {
            @Override
            public void run() {
                List<String> logs = LogHelper.getLogLines();
                uploadLog(logs);
                mBgHandler.postDelayed(this, 10000L);
            }
        };
    }

    public void start() {
        mBgHandler.postDelayed(mRunnable, 10000L);
    }

    public void release() {
        mBgHandler.removeCallbacksAndMessages(null);
    }

    private void uploadLog(final List<String> logs) {
        if (logs == null || logs.size() == 0) {
            return;
        }
        for (String log: logs ) {
            Log.e("MY_DEBUG", log);
        }
        RequestManager.uploadLog(mRunId, mEdgeId, logs, new OnLogUploadListener() {
            private int retryCnt = 3;

            @Override
            public void onLogUploaded(boolean success) {
                if (!success && retryCnt < 0) {
                    retryCnt--;
                    RequestManager.uploadLog(mRunId, mEdgeId, logs, this);
                }
            }
        });
        MetricsReporter.getInstance().reportSystemMetric(mRunId, mEdgeId);
    }
}
