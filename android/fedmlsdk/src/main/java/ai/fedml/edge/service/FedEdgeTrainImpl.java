package ai.fedml.edge.service;

import android.content.Context;

import ai.fedml.edge.OnTrainProgressListener;
import ai.fedml.edge.OnTrainingStatusListener;
import ai.fedml.edge.service.entity.TrainProgress;
import ai.fedml.edge.utils.LogHelper;
import ai.fedml.edge.utils.preference.SharePreferencesData;

import androidx.annotation.NonNull;

final class FedEdgeTrainImpl implements FedEdgeTrainApi {
    private ClientAgentManager mClientAgent;
    private volatile String mBindEdgeId = null;

    private OnTrainingStatusListener mOnTrainingStatusListener;
    private OnTrainProgressListener mOnTrainProgressListener;

    public FedEdgeTrainImpl() {
    }

    @Override
    public void init(@NonNull Context context, @NonNull final OnTrainingStatusListener onTrainingStatusListener,
                     @NonNull final OnTrainProgressListener onTrainProgressListener) {
        ContextHolder.initialize(context);
        mOnTrainingStatusListener = onTrainingStatusListener;
        mOnTrainProgressListener = onTrainProgressListener;

        // EdgeService process may restart after exception, so we should init ClientAgentManager with cached BindEdgeID
        mBindEdgeId = SharePreferencesData.getBindingId();
        if (mBindEdgeId != null && !mBindEdgeId.equals("")) {
            Initializer.getInstance().initial(() -> {
                LogHelper.i("FedMLDebug. OnCreate() mBindEdgeId = " + mBindEdgeId);
                if (mClientAgent == null) {
                    mClientAgent = new ClientAgentManager(mBindEdgeId, mOnTrainingStatusListener, mOnTrainProgressListener);
                    mClientAgent.start();
                    LogHelper.i("FedMLDebug. init() ClientAgentManager started");
                }
            });
        }
    }

    @Override
    public void bindEdge(String bindEdgeId) {
        mBindEdgeId = bindEdgeId;
        Initializer.getInstance().initial(() -> {
            LogHelper.i("FedMLDebug. bindEdge() mBindEdgeId = " + mBindEdgeId);
            if (mClientAgent == null) {
                mClientAgent = new ClientAgentManager(mBindEdgeId, mOnTrainingStatusListener, mOnTrainProgressListener);
                mClientAgent.start();
                LogHelper.i("FedMLDebug. bindEdge() ClientAgentManager started");
            }
        });
    }

    @Override
    public int getTrainStatus() {
        return 0;
    }


    @Override
    public TrainProgress getTrainProgress() {
        return null;
    }
}
