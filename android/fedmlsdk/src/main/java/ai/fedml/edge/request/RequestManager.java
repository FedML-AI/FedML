package ai.fedml.edge.request;

import ai.fedml.edge.request.listener.OnConfigListener;
import ai.fedml.edge.request.listener.OnLogUploadListener;
import ai.fedml.edge.request.parameter.BindingAccountReq;
import ai.fedml.edge.request.parameter.ConfigReq;
import ai.fedml.edge.request.parameter.LogUploadReq;

import ai.fedml.edge.request.response.ConfigResponse;
import ai.fedml.edge.utils.DeviceUtils;
import androidx.annotation.NonNull;

import java.io.IOException;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.TimeUnit;

import ai.fedml.edge.BuildConfig;
import ai.fedml.edge.request.listener.OnBindingListener;
import ai.fedml.edge.request.listener.OnUnboundListener;
import ai.fedml.edge.request.listener.OnUserInfoListener;
import ai.fedml.edge.request.response.BaseResponse;
import ai.fedml.edge.request.response.BindingResponse;
import ai.fedml.edge.request.response.TokenResponse;
import ai.fedml.edge.request.response.UserInfoResponse;
import ai.fedml.edge.utils.LogHelper;
import ai.fedml.edge.utils.preference.SharePreferencesData;
import okhttp3.OkHttpClient;
import okhttp3.logging.HttpLoggingInterceptor;
import okhttp3.logging.HttpLoggingInterceptor.Level;
import retrofit2.Call;
import retrofit2.Callback;
import retrofit2.Response;
import retrofit2.Retrofit;
import retrofit2.converter.gson.GsonConverterFactory;

public final class RequestManager {
    private static final String BASE_API_SERVER_URL = BuildConfig.MLOPS_SVR;
    private static Retrofit retrofit;
    private static final Map<Class<?>, Object> serviceMap = new ConcurrentHashMap<>(4);

    private static Retrofit retrofit() {
        if (retrofit == null) {
            HttpLoggingInterceptor loggingInterceptor = new HttpLoggingInterceptor().setLevel(Level.NONE);

            OkHttpClient okHttpClient = new OkHttpClient.Builder()
                    .writeTimeout(30_1000, TimeUnit.MILLISECONDS)
                    .readTimeout(20_1000, TimeUnit.MILLISECONDS)
                    .connectTimeout(15_1000, TimeUnit.MILLISECONDS)
                    .addInterceptor(loggingInterceptor)
                    .build();

            retrofit = new Retrofit.Builder()
                    .baseUrl(BASE_API_SERVER_URL)
                    .addConverterFactory(GsonConverterFactory.create())
                    .client(okHttpClient)
                    .build();
        }
        return retrofit;
    }

    @SuppressWarnings("unchecked")
    private static <T> T getService(final Class<T> service) {
        Object myService = serviceMap.get(service);
        if (myService == null) {
            myService = retrofit().create(service);
            serviceMap.put(service, myService);
        }
        return (T) myService;
    }

    public static String getAccessToken(final String groupId, String edgeId) {
        Call<TokenResponse> call = getService(AccessTokenService.class).getAccessToken(groupId, edgeId);
        Response<TokenResponse> response;
        try {
            response = call.execute();
        } catch (IOException e) {
            LogHelper.e(e, "getAccessToken req failed.");
            return "";
        }
        return response.body() != null ? response.body().getData() : "";
    }

    public static void bindingAccount(@NonNull BindingAccountReq req, @NonNull OnBindingListener listener) {
        Call<BindingResponse> call = getService(UserManagerService.class).bindingAccount(req);
        call.enqueue(new Callback<BindingResponse>() {
            @Override
            public void onResponse(@NonNull Call<BindingResponse> call, @NonNull final Response<BindingResponse> response) {
                LogHelper.d("bindingAccount onResponse " + response);
                if (response.body() == null) {
                    listener.onDeviceBinding(null);
                } else {
                    BindingResponse.BindingData bindingData = response.body().getBindingData();
                    if (bindingData != null) {
                        SharePreferencesData.saveBindingId(bindingData.getBindingId());
                        SharePreferencesData.saveAccountId(req.getAccountId());
                    }
                    listener.onDeviceBinding(bindingData);
                }
            }

            @Override
            public void onFailure(@NonNull Call<BindingResponse> call, @NonNull Throwable t) {
                LogHelper.e(t, "bindingAccount onFailure");
                listener.onDeviceBinding(null);
            }
        });
    }

    public static void unboundAccount(@NonNull final String edgeId, @NonNull final OnUnboundListener listener) {
        Call<BaseResponse> call = getService(UserManagerService.class).unbindingAccount(edgeId);
        call.enqueue(new Callback<BaseResponse>() {
            @Override
            public void onResponse(@NonNull Call<BaseResponse> call, @NonNull Response<BaseResponse> response) {
                if (response.body() == null) {
                    listener.onUnbound(false);
                    return;
                }
                SharePreferencesData.deleteBindingId();
                listener.onUnbound(true);
            }

            @Override
            public void onFailure(@NonNull Call<BaseResponse> call, @NonNull Throwable t) {
                LogHelper.e(t, "===onFailure=== ");
                listener.onUnbound(false);
            }
        });
    }

    public static void getUserInfo(@NonNull final OnUserInfoListener listener) {
        final String deviceId = DeviceUtils.getDeviceId();
        Call<UserInfoResponse> call = getService(UserManagerService.class).getUserInfo(deviceId);
        call.enqueue(new Callback<UserInfoResponse>() {
            @Override
            public void onResponse(@NonNull Call<UserInfoResponse> call, @NonNull Response<UserInfoResponse> response) {
                LogHelper.d("getUserInfo onResponse=== " + response);
                UserInfoResponse userInfo = response.body();
                if (userInfo == null) {
                    listener.onGetUserInfo(null);
                    return;
                }
                if (userInfo.getData() == null || userInfo.getData().size() <= 0) {
                    listener.onGetUserInfo(null);
                    return;
                }
                listener.onGetUserInfo(userInfo.getData().get(0));
            }

            @Override
            public void onFailure(@NonNull Call<UserInfoResponse> call, @NonNull Throwable t) {
                LogHelper.e(t, "getUserInfo failed.");
                listener.onGetUserInfo(null);
            }
        });
    }

    public static void uploadLog(final long runId, final long edgeId, final List<String> logLines,
                                 @NonNull final OnLogUploadListener listener) {
        long currentTimeSecond = System.currentTimeMillis() / 1000;
        LogUploadReq req = LogUploadReq.builder()
                .runId(runId).edgeId(edgeId).logLines(logLines)
                .createTime(currentTimeSecond).updateTime(currentTimeSecond)
                .createdBy(String.valueOf(edgeId)).updatedBy(String.valueOf(edgeId))
                .build();
        Call<BaseResponse> call = getService(LogService.class).logUpload(req);
        call.enqueue(new Callback<BaseResponse>() {
            @Override
            public void onResponse(@NonNull Call<BaseResponse> call, @NonNull Response<BaseResponse> response) {
                listener.onLogUploaded(response.isSuccessful());
            }

            @Override
            public void onFailure(@NonNull Call<BaseResponse> call, @NonNull Throwable t) {
                LogHelper.e(t, "uploadLog failed.");
                listener.onLogUploaded(false);
            }
        });
    }

    public static void fetchConfig(@NonNull final OnConfigListener listener) {
        Call<ConfigResponse> call = getService(UserManagerService.class).fetchConfig(new ConfigReq());
        call.enqueue(new Callback<ConfigResponse>() {
            @Override
            public void onResponse(@NonNull Call<ConfigResponse> call, @NonNull Response<ConfigResponse> response) {
                ConfigResponse resp = response.body();
                if (resp == null || resp.getData() == null) {
                    listener.onConfig(null);
                    return;
                }
                listener.onConfig(resp.getData());
            }

            @Override
            public void onFailure(@NonNull Call<ConfigResponse> call, @NonNull Throwable t) {
                LogHelper.e(t, "getUserInfo failed.");
                listener.onConfig(null);
            }
        });
    }
}
