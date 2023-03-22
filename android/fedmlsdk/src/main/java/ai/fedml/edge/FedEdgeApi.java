package ai.fedml.edge;

import android.content.Context;

public interface FedEdgeApi {
    void init(Context appContext);

    String getBoundEdgeId();

    void bindEdge(String bindId);

    void train();

    void getTrainingStatus();

    void getEpochAndLoss();

    void setTrainingStatusListener(OnTrainingStatusListener listener);

    void setEpochLossListener(OnTrainProgressListener listener);

    String getHyperParameters();

    void setPrivatePath(final String path);

    String getPrivatePath();
}
