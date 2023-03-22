package ai.fedml.edge.service.entity;

import ai.fedml.edge.service.component.OnTrainCompletedListener;
import lombok.Builder;
import lombok.Data;

@Builder
@Data
public class TrainingParams {
    private String trainModelPath;
    private long edgeId;
    private long runId;
    private int clientIdx;
    private int clientRound;
    /**
     * Dataset type should be matched with dataset. 0: mnist dataset, 1: cifar10 dataset.
     */
    private String dataSet;
    private int batchSize;
    private double learningRate;
    private int epochNum;
    private int trainSize;
    private int testSize;
    private OnTrainCompletedListener listener;
}
