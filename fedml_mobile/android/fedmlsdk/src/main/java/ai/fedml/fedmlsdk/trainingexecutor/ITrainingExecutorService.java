package ai.fedml.fedmlsdk.trainingexecutor;

import okhttp3.MultipartBody;
import okhttp3.RequestBody;
import okhttp3.ResponseBody;
import retrofit2.Call;
import retrofit2.http.Body;
import retrofit2.http.Field;
import retrofit2.http.FormUrlEncoded;
import retrofit2.http.GET;
import retrofit2.http.Headers;
import retrofit2.http.Multipart;
import retrofit2.http.POST;
import retrofit2.http.Part;
import retrofit2.http.Streaming;
import retrofit2.http.Url;

public interface ITrainingExecutorService {
    @POST("/api/register")
    @FormUrlEncoded
    Call<ExecutorResponse> registerDevice(@Field("device_id") String deviceId);

    @POST("api/deviceOnLine")
    @Headers("Authorization: application/json")
    Call<ResponseBody> deviceOnLine(@Body RequestBody req);

    @POST("api/upload")
    @Multipart
    Call<ResponseBody> upload(@Part("filename") RequestBody filename, @Part MultipartBody.Part file);

    @Streaming
    @GET
    Call<ResponseBody> downloadFile(@Url String fileUrl);
}
