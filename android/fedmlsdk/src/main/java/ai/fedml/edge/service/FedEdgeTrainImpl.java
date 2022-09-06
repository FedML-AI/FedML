package ai.fedml.edge.service;

import android.content.Context;
import android.text.TextUtils;

import ai.fedml.edge.OnTrainProgressListener;
import ai.fedml.edge.OnTrainingStatusListener;
import ai.fedml.edge.service.entity.TrainProgress;
import ai.fedml.edge.utils.LogHelper;
import androidx.annotation.NonNull;

class FedEdgeTrainImpl implements FedEdgeTrainApi {
    private ClientAgentManager mClientAgent;
    private volatile boolean isBound = false;
    private volatile String mBindEdgeId;

    public FedEdgeTrainImpl() {
    }

    @Override
    public void init(@NonNull Context context, @NonNull final OnTrainingStatusListener onTrainingStatusListener,
                     @NonNull final OnTrainProgressListener onTrainProgressListener) {
        ContextHolder.initialize(context);
        Initializer.getInstance().initial(() -> {
            LogHelper.d("Initializer initial finished");
            mClientAgent = new ClientAgentManager(onTrainingStatusListener, onTrainProgressListener);
            if (!isBound && !TextUtils.isEmpty(mBindEdgeId)) {
                mClientAgent.bindCommunicator(mBindEdgeId);
            }
        });
    }

    @Override
    public void bindEdge(String bindEdgeId) {
        mBindEdgeId = bindEdgeId;
        if (mClientAgent != null) {
            LogHelper.d("bindEdge finished");
            mClientAgent.bindCommunicator(bindEdgeId);
            isBound = true;
        }
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
