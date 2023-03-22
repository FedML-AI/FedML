package ai.fedml.edge.request.response;

import com.google.gson.annotations.SerializedName;

import java.util.List;

import ai.fedml.edge.request.response.BaseResponse;
import lombok.Data;
import lombok.EqualsAndHashCode;
import lombok.ToString;


@Data
@EqualsAndHashCode(callSuper = true)
@ToString(callSuper = true)
public class UserInfoResponse extends BaseResponse {
    @SerializedName("data")
    private List<UserInfo> data;

    @Data
    public static class UserInfo {
        @SerializedName("id")
        private Integer id;
        @SerializedName("account")
        private String account;
        @SerializedName("friendids")
        private String friendIds;
        @SerializedName("phoneNumber")
        private String phoneNumber;
        @SerializedName("email")
        private String email;
        @SerializedName("firtname")
        private String firstName;
        @SerializedName("lastname")
        private String lastname;
        @SerializedName("company")
        private String company;
        @SerializedName("interest")
        private String interest;
        @SerializedName("createtime")
        private String createTime;
        @SerializedName("lastupdatetime")
        private String lastUpdateTime;
        @SerializedName("creater")
        private String creator;
        @SerializedName("userType")
        private String userType;
        @SerializedName("avatar")
        private String avatar;
        @SerializedName("lastToken")
        private String lastToken;
    }

}
