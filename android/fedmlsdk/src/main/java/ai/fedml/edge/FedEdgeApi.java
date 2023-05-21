package ai.fedml.edge;

import android.content.Context;

public interface FedEdgeApi {
    /**
     * Init
     */
    void init(Context appContext);

    /**
     * Edge ID
     */
    String getBoundEdgeId();

    void bindEdge(String bindId);

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
}
