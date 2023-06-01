package ai.fedml.edge.request;

import ai.fedml.edge.constants.UrlPath;
import ai.fedml.edge.request.listener.OnConfigListener;
import ai.fedml.edge.request.listener.OnLogUploadListener;
import ai.fedml.edge.request.parameter.BindingAccountReq;
import ai.fedml.edge.request.parameter.ConfigReq;
import ai.fedml.edge.request.parameter.EdgesError;
import ai.fedml.edge.request.parameter.LogUploadReq;

import ai.fedml.edge.request.response.ConfigResponse;
import ai.fedml.edge.request.response.UnBindingResponse;
import ai.fedml.edge.utils.DeviceUtils;

import androidx.annotation.NonNull;

import com.google.gson.Gson;

import java.util.HashMap;
import java.util.List;
import java.util.Map;

import ai.fedml.edge.BuildConfig;
import ai.fedml.edge.request.listener.OnBindingListener;
import ai.fedml.edge.request.listener.OnUnboundListener;
import ai.fedml.edge.request.listener.OnUserInfoListener;
import ai.fedml.edge.request.response.BaseResponse;
import ai.fedml.edge.request.response.BindingResponse;
import ai.fedml.edge.request.response.UserInfoResponse;
import ai.fedml.edge.utils.HttpUtils;
import ai.fedml.edge.utils.LogHelper;
import ai.fedml.edge.utils.preference.SharePreferencesData;

public final class RequestManager {
    private static final String BASE_API_SERVER_URL = BuildConfig.MLOPS_SVR;

    public static void bindingAccount(@NonNull BindingAccountReq req, @NonNull OnBindingListener listener) {
        Gson gson = new Gson();
        String json = gson.toJson(req);

        HttpUtils.doPost(BASE_API_SERVER_URL + UrlPath.PATH_EDGE_BINDING, json, new HttpUtils.CallBack() {
            @Override
            public void onRequestComplete(String result) {
                LogHelper.i("bindingAccount result: " + result);
                BindingResponse response = gson.fromJson(result, BindingResponse.class);
                if (response == null || !"SUCCESS".equals(response.getCode()) || response.getBindingData() == null) {
                    LogHelper.w("bindingAccount response is null or not SUCCESS");
                    listener.onDeviceBinding(null);
                    return;
                }

                BindingResponse.BindingData bindingData = response.getBindingData();
                if (bindingData != null) {
                    SharePreferencesData.saveBindingId(bindingData.getBindingId());
                    SharePreferencesData.saveAccountId(req.getAccountId());
                }
                listener.onDeviceBinding(bindingData);
            }

            @Override
            public void onRequestFailed() {
                LogHelper.w("bindingAccount onRequestFailed");
                listener.onDeviceBinding(null);
            }
        });
    }

    public static void unboundAccount(@NonNull final String edgeId, @NonNull final OnUnboundListener listener) {
        Map<String, String> params = new HashMap<>();
        params.put("id", edgeId);
        HttpUtils.doPost(BASE_API_SERVER_URL + UrlPath.PATH_EDGE_UNBINDING, params, new HttpUtils.CallBack() {
            @Override
            public void onRequestComplete(String result) {
                LogHelper.i("unboundAccount result: " + result);
                Gson gson = new Gson();
                UnBindingResponse response = gson.fromJson(result, UnBindingResponse.class);
                if (response == null || !"SUCCESS".equals(response.getCode()) || response.getData() == null) {
                    LogHelper.w("unboundAccount response is null");
                    listener.onUnbound(false);
                    return;
                }

                SharePreferencesData.deleteBindingId();
                listener.onUnbound(true);
            }

            @Override
            public void onRequestFailed() {
                LogHelper.w("unboundAccount onRequestFailed");
                listener.onUnbound(false);
            }
        });
    }

    public static void getUserInfo(@NonNull final OnUserInfoListener listener) {
        final String deviceId = DeviceUtils.getDeviceId();
        Map<String, String> params = new HashMap<>();
        params.put("id", deviceId);
        HttpUtils.doGet(BASE_API_SERVER_URL + UrlPath.PATH_USER_INFO, params, new HttpUtils.CallBack() {
            @Override
            public void onRequestComplete(String result) {
                LogHelper.i("getUserInfo result: " + result);
                Gson gson = new Gson();
                UserInfoResponse response = gson.fromJson(result, UserInfoResponse.class);
                if (response == null || !"SUCCESS".equals(response.getCode()) || response.getData() == null) {
                    LogHelper.d("getUserInfo response is null or not SUCCESS");
                    listener.onGetUserInfo(null);
                    return;
                }
                listener.onGetUserInfo(response.getData().get(0));
            }

            @Override
            public void onRequestFailed() {
                LogHelper.w("getUserInfo onRequestFailed");
                listener.onGetUserInfo(null);
            }
        });
    }

    public static void uploadLog(final long runId, final long edgeId, final List<String> logLines,
                                 final List<EdgesError> errorLines, @NonNull final OnLogUploadListener listener) {
        long currentTimeSecond = System.currentTimeMillis() / 1000;
        LogUploadReq req = LogUploadReq.builder()
                .runId(runId).edgeId(edgeId).logLines(logLines).errorLines(errorLines)
                .createTime(currentTimeSecond).updateTime(currentTimeSecond)
                .createdBy(String.valueOf(edgeId)).updatedBy(String.valueOf(edgeId))
                .build();
        Gson gson = new Gson();
        String json = gson.toJson(req);
        HttpUtils.doPost(BASE_API_SERVER_URL + UrlPath.PATH_LOG_UPLOAD, json, new HttpUtils.CallBack() {

            @Override
            public void onRequestComplete(String result) {
                LogHelper.d("uploadLog result: " + result);
                BaseResponse response = gson.fromJson(result, BaseResponse.class);
                if (null == response || !"SUCCESS".equals(response.getCode())) {
                    LogHelper.w("uploadLog response is null or not SUCCESS");
                    listener.onLogUploaded(false);
                    return;
                }
                listener.onLogUploaded(true);
            }

            @Override
            public void onRequestFailed() {
                LogHelper.w("uploadLog failed");
                listener.onLogUploaded(false);
            }
        });
    }

    public static void fetchConfig(@NonNull final OnConfigListener listener) {
        ConfigReq req = new ConfigReq();
        Gson gson = new Gson();
        String json = gson.toJson(req);

        HttpUtils.doPost(BASE_API_SERVER_URL + UrlPath.PATH_FETCH_CONFIG, json, new HttpUtils.CallBack() {
            @Override
            public void onRequestComplete(String result) {
                LogHelper.i("fetchConfig result: " + result);
                ConfigResponse response = gson.fromJson(result, ConfigResponse.class);
                if (response == null || response.getData() == null) {
                    listener.onConfig(null);
                    return;
                }
                listener.onConfig(response.getData());
            }

            @Override
            public void onRequestFailed() {
                LogHelper.w("fetchConfig failed");
                listener.onConfig(null);
            }
        });
    }

}
