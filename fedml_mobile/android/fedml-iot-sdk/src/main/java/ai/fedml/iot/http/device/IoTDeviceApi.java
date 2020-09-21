package ai.fedml.iot.http.device;

import ai.fedml.iot.http.CommonResult;

import retrofit2.Call;
import retrofit2.http.Field;
import retrofit2.http.FormUrlEncoded;
import retrofit2.http.GET;
import retrofit2.http.POST;
import retrofit2.http.Path;

public interface IoTDeviceApi {
    @GET("iotdevice/{IOT_UUID}")
    Call<ResultDeviceInfo<DeviceInfo>> getDeviceInfo(@Path("IOT_UUID") String IOT_UUID);

    @FormUrlEncoded
    @POST("iotdevice")
    Call<CommonResult> registerDevice(@Field("deviceinfo") String jsonDeviceinfo);
}
