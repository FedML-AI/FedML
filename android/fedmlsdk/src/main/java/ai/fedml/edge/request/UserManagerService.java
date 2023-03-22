package ai.fedml.edge.request;

import ai.fedml.edge.request.parameter.BindingAccountReq;
import ai.fedml.edge.request.parameter.ConfigReq;
import ai.fedml.edge.request.response.BaseResponse;
import ai.fedml.edge.request.response.BindingResponse;
import ai.fedml.edge.request.response.ConfigResponse;
import ai.fedml.edge.request.response.UserInfoResponse;
import retrofit2.Call;
import retrofit2.http.Body;
import retrofit2.http.GET;
import retrofit2.http.Headers;
import retrofit2.http.POST;
import retrofit2.http.Query;

public interface UserManagerService {
    @GET("/fedmlOpsServer/edges/device")
    Call<UserInfoResponse> getUserInfo(@Query("id") String edgeId);

    @POST("/fedmlOpsServer/edges/binding")
    @Headers("Content-Type: application/json;charset=UTF-8")
    Call<BindingResponse> bindingAccount(@Body BindingAccountReq req);

    @POST("/fedmlOpsServer/edges/unbound")
    @Headers("Content-Type: application/json;charset=UTF-8")
    Call<BaseResponse> unbindingAccount(@Query("id") String bindingId);

    @POST("/fedmlOpsServer/configs/fetch")
    @Headers("Content-Type: application/json;charset=UTF-8")
    Call<ConfigResponse> fetchConfig(@Body ConfigReq req);
}
