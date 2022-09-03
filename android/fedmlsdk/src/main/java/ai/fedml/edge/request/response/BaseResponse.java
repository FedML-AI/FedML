package ai.fedml.edge.request.response;

import com.google.gson.annotations.SerializedName;

import lombok.Data;
import lombok.ToString;

@Data
@ToString
public class BaseResponse {
    @SerializedName("code")
    private String code;
    @SerializedName("message")
    private String message;
}
