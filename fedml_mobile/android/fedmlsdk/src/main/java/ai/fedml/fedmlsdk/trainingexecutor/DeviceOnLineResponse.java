package ai.fedml.fedmlsdk.trainingexecutor;

import com.google.gson.annotations.SerializedName;

import lombok.Data;

@Data
public class DeviceOnLineResponse {
    @SerializedName("errno")
    private int errno;
    @SerializedName("executorId")
    private String executorId;
    @SerializedName("executorTopic")
    private String executorTopic;
}
