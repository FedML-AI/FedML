package ai.fedml.edge.service.entity;

import lombok.Builder;
import lombok.Data;

@Data
@Builder
public class TrainProgress {
    private int epoch;
    private float loss;
    private float accuracy;
    private int progress;
}
