package ai.fedml.edge.service.communicator.message;

import com.google.gson.annotations.SerializedName;

import lombok.Data;
import lombok.EqualsAndHashCode;
import lombok.experimental.SuperBuilder;

@EqualsAndHashCode(callSuper = true)
@Data
@SuperBuilder
public class BackModelMessage extends BaseMessage {
    @SerializedName("model_params")
    private String modelParams;

    @SerializedName("client_idx")
    private String clientIdx;

    @SerializedName("num_samples")
    private long numSamples;
}
