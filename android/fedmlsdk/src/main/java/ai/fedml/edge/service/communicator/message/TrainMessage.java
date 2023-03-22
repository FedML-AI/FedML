package ai.fedml.edge.service.communicator.message;

import com.google.gson.annotations.SerializedName;

import lombok.Data;
import lombok.EqualsAndHashCode;
import lombok.experimental.SuperBuilder;

@EqualsAndHashCode(callSuper = true)
@Data
@SuperBuilder
public class TrainMessage extends BaseMessage {
    @SerializedName("model_params")
    private String modelParams;

    @SerializedName("client_idx")
    private String clientIdx;
}
