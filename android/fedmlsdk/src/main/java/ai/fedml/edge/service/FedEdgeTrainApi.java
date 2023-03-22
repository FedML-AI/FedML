package ai.fedml.edge.service;

import android.content.Context;

import ai.fedml.edge.OnAccuracyLossListener;
import ai.fedml.edge.OnTrainProgressListener;
import ai.fedml.edge.OnTrainingStatusListener;
import ai.fedml.edge.service.entity.TrainProgress;

import androidx.annotation.NonNull;

public interface FedEdgeTrainApi {
    /**
     * initialize FedMl Mobile Device
     *
     * @param context Context
     */
    void init(@NonNull final Context context, @NonNull final OnTrainingStatusListener onTrainingStatusListener,
              @NonNull final OnTrainProgressListener onTrainProgressListener);

    /**
     * bind MQTT Communicator
     *
     * @param bindEdgeId bindId
     */
    void bindEdge(final String bindEdgeId);

    /**
     * get the train status
     *
     * @return status
     */
    int getTrainStatus();

    TrainProgress getTrainProgress();
}
