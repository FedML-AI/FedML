package ai.fedml.edge.service;

import android.content.Context;

import ai.fedml.edge.OnTrainProgressListener;
import ai.fedml.edge.OnTrainingStatusListener;
import ai.fedml.edge.service.entity.TrainProgress;
import ai.fedml.edge.utils.LogHelper;
import androidx.annotation.NonNull;

class FedEdgeTrainImpl implements FedEdgeTrainApi {
    private ClientAgentManager mClientAgent;
    private volatile String mBindEdgeId;

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
    }

    @Override
    public void bindEdge(String bindEdgeId) {
        LogHelper.d("FedMLDebug. bindEdge(), bindEdgeId = " + bindEdgeId);
        mBindEdgeId = bindEdgeId;
        Initializer.getInstance().initial(() -> {
            LogHelper.d("FedMLDebug. Initializer initial finished. mBindEdgeId = " + mBindEdgeId);
            mClientAgent = new ClientAgentManager(mBindEdgeId, mOnTrainingStatusListener, mOnTrainProgressListener);
            mClientAgent.start();
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
