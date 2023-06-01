package ai.fedml.edge.request.parameter;

import com.google.gson.annotations.SerializedName;

import lombok.Builder;
import lombok.Data;

@Data
@Builder
public class BindingAccountReq {
    @SerializedName("accountid")
    private String accountId;

    @SerializedName("deviceid")
    private String deviceId;

    @SerializedName("type")
    @Builder.Default
    private final String type = "Android";

    @SerializedName("state")
    @Builder.Default
    private final String state = "IDLE";

    @SerializedName("role")
    @Builder.Default
    private final String role = "client";

    @SerializedName("core_type")
    private String cpuAbi;

    @SerializedName("os_ver")
    private String osVersion;

    @SerializedName("memory")
    private String memory;

    public String getAccountId() {
        return accountId;
    }
}
