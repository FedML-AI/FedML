package ai.fedml.edge.service.communicator.message;

import com.google.gson.annotations.SerializedName;

import lombok.Data;
import lombok.EqualsAndHashCode;
import lombok.experimental.SuperBuilder;

@EqualsAndHashCode(callSuper = true)
@Data
@SuperBuilder
public class TrainStatusMessage extends BaseMessage {

    @SerializedName("client_status")
    private String status;

    @SerializedName("client_os")
    private String os;
}
