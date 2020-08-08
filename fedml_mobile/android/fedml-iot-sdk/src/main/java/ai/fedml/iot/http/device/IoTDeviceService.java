package ai.fedml.iot.http.device;

import ai.fedml.iot.Config;
import ai.fedml.iot.http.CommonResult;
import ai.fedml.iot.utils.LogUtils;

import retrofit2.Call;
import retrofit2.Callback;
import retrofit2.Response;
import retrofit2.Retrofit;
import retrofit2.converter.gson.GsonConverterFactory;

public class IoTDeviceService {
    private String TAG = Config.COMMON_TAG + getClass().getSimpleName();

    private static volatile IoTDeviceService ioTAppService = null;
    private Retrofit retrofit = null;

    private IoTDeviceApi mIoTDeviceApi = null;

    private IoTDeviceService() {
        retrofit = new Retrofit.Builder()
                // 设置BaseURL
                .baseUrl(Config.BASE_URL_IOT_APP_SERVICE)
                // 增加返回值为Gson的支持(以实体类返回)
                .addConverterFactory(GsonConverterFactory.create())
                .build();

        mIoTDeviceApi = retrofit.create(IoTDeviceApi.class);
    }

    public static IoTDeviceService getInstance() {
        if (ioTAppService == null) {
            synchronized (IoTDeviceService.class) {
                if (ioTAppService == null) {
                    ioTAppService = new IoTDeviceService();
                }
            }
        }
        return ioTAppService;
    }

    private IoTDeviceApi getIoTDeviceApi() {
        return mIoTDeviceApi;
    }

    public interface DeviceInfoCallback {
        void onSuccess(DeviceInfo deviceInfo);

        void onFail(int errorCode, String errorMsg);
    }

    public void getDeviceInfo(String strIoTUUID, final DeviceInfoCallback callback) {
        IoTDeviceApi ioTDeviceApi = getIoTDeviceApi();
        Call<ResultDeviceInfo<DeviceInfo>> call = ioTDeviceApi.getDeviceInfo(strIoTUUID);
        call.enqueue(new Callback<ResultDeviceInfo<DeviceInfo>>() {
            @Override
            public void onResponse(Call<ResultDeviceInfo<DeviceInfo>> call,
                                   Response<ResultDeviceInfo<DeviceInfo>> response) {
                try{
                    if(response == null) return;
                    ResultDeviceInfo<DeviceInfo> resultDeviceInfo = response.body();
                    if(resultDeviceInfo == null){
                        if (callback != null) {
                            callback.onFail(CommonResult.ERROR_CODE_SERVERERR, "Server Error!");
                        }
                        return;
                    }
                    if (resultDeviceInfo.getErrorCode() == CommonResult.ERROR_CODE_SUCCESS) {
                        if (callback != null) {
                            callback.onSuccess(resultDeviceInfo.getDeviceInfo());
                        }
                    } else {
                        if (callback != null) {
                            callback.onFail(resultDeviceInfo.getErrorCode(), resultDeviceInfo.getErrorMsg());
                        }
                    }
                }catch (Exception e){
                    e.printStackTrace();
                    LogUtils.e(TAG, "server down!");
                    if (callback != null) {
                        callback.onFail(CommonResult.ERROR_CODE_SERVERERR, "Server Error!");
                    }
                }

            }

            @Override
            public void onFailure(Call<ResultDeviceInfo<DeviceInfo>> call, Throwable throwable) {
                if (callback != null) {
                    throwable.printStackTrace();
                    callback.onFail(CommonResult.ERROR_CODE_NETERR, "http onFailure");
                }
            }
        });
    }
}