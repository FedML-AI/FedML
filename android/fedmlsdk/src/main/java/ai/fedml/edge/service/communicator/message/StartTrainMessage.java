
package ai.fedml.edge.service.communicator.message;

import com.google.gson.annotations.SerializedName;

import java.util.List;

import lombok.Data;

@Data
public class StartTrainMessage {

    private Double clientLearningRate;

    private Long clientNumPerRound;

    private String clientOptimizer;

    private Long commRound;

    private String communicationBackend;

    private String dataset;

    @SerializedName("edgeids")
    private List<Long> edgeIds;

    @SerializedName("groupid")
    private String groupId;

    private Long id;

    private Long localEpoch;

    private String modelName;

    private String name;

    private String partitionMethod;

    @SerializedName("projectid")
    private String projectId;

    private Long runId;

    @SerializedName("starttime")
    private Long startTime;

    private String timestamp;

    private String token;

    private Long trainBatchSize;

    private List<String> urls;

    @SerializedName("userids")
    private List<String> userIds;
}
