package ai.fedml.edge.request;

import ai.fedml.edge.request.parameter.LogUploadReq;
import ai.fedml.edge.request.response.BaseResponse;
import retrofit2.Call;
import retrofit2.http.Body;
import retrofit2.http.Headers;
import retrofit2.http.POST;

public interface LogService {
    @POST("/fedmlLogsServer/logs/update")
    @Headers("Content-Type: application/json;charset=UTF-8")
    Call<BaseResponse> logUpload(@Body LogUploadReq req);
}
