package ai.fedml.edge.service.communicator.message;

import com.google.gson.annotations.SerializedName;

import lombok.Data;
import lombok.experimental.SuperBuilder;

@Data
@SuperBuilder
public class BaseMessage implements MessageDefine {

    @SerializedName("operation")
    private String operation;

    @SerializedName("msg_type")
    private int messageType;

    @SerializedName("sender")
    private long sender;

    @SerializedName("receiver")
    private long receiver;
}
