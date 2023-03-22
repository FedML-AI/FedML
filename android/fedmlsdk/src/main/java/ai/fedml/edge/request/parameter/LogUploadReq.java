package ai.fedml.edge.request.parameter;

import com.google.gson.annotations.SerializedName;

import java.util.List;

import lombok.Builder;
import lombok.Data;

@Data
@Builder
public class LogUploadReq {
    @SerializedName("run_id")
    private Long runId;

    @SerializedName("edge_id")
    private Long edgeId;

    @SerializedName("logs")
    private List<String> logLines;

    /**
     * create time in seconds since the Epoch.
     */
    @SerializedName("create_time")
    private Long createTime;

    /**
     * update time in seconds since the Epoch.
     */
    @SerializedName("update_time")
    private Long updateTime;

    @SerializedName("created_by")
    private String createdBy;

    @SerializedName("updated_by")
    private String updatedBy;
}
