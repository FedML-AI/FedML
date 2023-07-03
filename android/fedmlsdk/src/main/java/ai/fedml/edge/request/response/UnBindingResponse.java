package ai.fedml.edge.request.response;

import com.google.gson.annotations.SerializedName;

import lombok.Data;
import lombok.EqualsAndHashCode;
import lombok.ToString;

@Data
@ToString(callSuper = true)
@EqualsAndHashCode(callSuper = true)
public class UnBindingResponse extends BaseResponse {

    @SerializedName("data")
    private String data;

}
