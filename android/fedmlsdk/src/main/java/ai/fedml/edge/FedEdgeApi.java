package ai.fedml.edge;

import android.content.Context;

import ai.fedml.edge.request.listener.OnBindingListener;
import ai.fedml.edge.request.listener.OnUnboundListener;
import ai.fedml.edge.request.listener.OnUserInfoListener;
import androidx.annotation.NonNull;

public interface FedEdgeApi {
    /**
     * Init
     */
    void init(Context appContext);

    void bindingAccount(@NonNull String accountId, @NonNull String deviceId, @NonNull OnBindingListener listener);

    void unboundAccount(@NonNull final String edgeId, @NonNull final OnUnboundListener listener);

    /**
     * Edge ID
     */
    String getBoundEdgeId();

    void bindEdge(String bindId);

    void getUserInfo(@NonNull final OnUserInfoListener listener);

    /**
     * Training
     */
    void train();

    void getTrainingStatus();

    void getEpochAndLoss();

    void setTrainingStatusListener(OnTrainingStatusListener listener);

    void setEpochLossListener(OnTrainProgressListener listener);

    String getHyperParameters();

    /**
     * Data
     */
    void setPrivatePath(final String path);

    String getPrivatePath();

    /**
     * unInit
     */
    void unInit();
}
