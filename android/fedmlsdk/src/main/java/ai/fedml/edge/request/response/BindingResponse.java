package ai.fedml.edge.request.response;

import com.google.gson.annotations.SerializedName;

import ai.fedml.edge.request.response.BaseResponse;
import lombok.Data;
import lombok.EqualsAndHashCode;
import lombok.ToString;


@Data
@EqualsAndHashCode(callSuper = true)
@ToString(callSuper = true)
public class BindingResponse extends BaseResponse {
    @SerializedName("data")
    private BindingData bindingData;

    @Data
    public static class BindingData {
        @SerializedName("id")
        private String bindingId;
    }
}
