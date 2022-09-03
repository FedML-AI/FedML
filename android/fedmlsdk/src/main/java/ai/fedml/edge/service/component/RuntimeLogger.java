package ai.fedml.edge.service.component;

import android.os.Handler;

import java.util.List;

import ai.fedml.edge.request.RequestManager;
import ai.fedml.edge.request.listener.OnLogUploadListener;
import ai.fedml.edge.utils.BackgroundHandler;
import ai.fedml.edge.utils.LogHelper;

public class RuntimeLogger {
    private final Handler mBgHandler;
    private final long mEdgeId;
    private final long mRunId;

    public RuntimeLogger(final long edgeId, final long runId) {
        mEdgeId = edgeId;
        mRunId = runId;
        mBgHandler = new BackgroundHandler("LogUploader");
    }

    public void initial() {
        mBgHandler.postDelayed(new Runnable() {
            @Override
            public void run() {
                List<String> logs = LogHelper.getLogLines();
                uploadLog(logs);
                mBgHandler.postDelayed(this, 10000L);
            }
        }, 10000L);
    }

    private void uploadLog(final List<String> logs) {
        if (logs == null || logs.size() == 0) {
            return;
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
