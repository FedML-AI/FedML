package ai.fedml.client.entity;

import com.google.gson.annotations.SerializedName;

import java.util.List;

import ai.fedml.edge.request.response.BaseResponse;
import lombok.Data;
import lombok.EqualsAndHashCode;
import lombok.ToString;


@Data
@EqualsAndHashCode(callSuper = true)
@ToString(callSuper = true)
public class VersionUpdateResponse extends BaseResponse {
    @SerializedName("data")
    private List<VersionUpdateInfo> versionList;

    @Data
    public static class VersionUpdateInfo {
        @SerializedName("createtime")
        private String createTime;
        @SerializedName("code")
        private Integer code;
        @SerializedName("name")
        private String name;
        @SerializedName("id")
        private Integer id;
        @SerializedName("url")
        private String url;
    }
}
