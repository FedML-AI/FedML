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

    public String getAccountId() {
        return accountId;
    }
}
