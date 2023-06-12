package ai.fedml.edge.service;

import ai.fedml.edge.request.RequestManager;
import ai.fedml.edge.request.response.ConfigResponse;

import androidx.annotation.NonNull;

import org.json.JSONException;

import ai.fedml.edge.utils.LogHelper;
import ai.fedml.edge.utils.TimeUtils;
import lombok.Getter;

@Getter
public class Initializer {
    public interface OnLoadListener {
        void onInitialized();
    }

    private static class LazyHolder {
        private final static Initializer S_INITIALIZER = new Initializer();
    }

    public static Initializer getInstance() {
        return LazyHolder.S_INITIALIZER;
    }

    private ConfigResponse.MqttConfig mqttConfig;

    private ConfigResponse.S3Config s3Config;

    private ConfigResponse.MlopsConfig mlopsConfig;

    public void initial(@NonNull final OnLoadListener listener) {
        RequestManager.fetchConfig(config -> {
            if (config == null) {
                throw new RuntimeException("fetch all Configs failed!");
            }
            mqttConfig = config.getMqttConfig();
            if (mqttConfig == null) {
                throw new RuntimeException("fetch mqtt config failed!");
            }
            s3Config = config.getS3Config();
            if (s3Config == null) {
                throw new RuntimeException("fetch s3 config failed!");
            }

            mlopsConfig = config.getMlopsConfig();
            if (mlopsConfig == null || mlopsConfig.getNtpResponse() == null) {
                throw new RuntimeException("fetch mlops config failed!");
            } else {
                ConfigResponse.NtpResponse ntpResponse = mlopsConfig.getNtpResponse();
                Long serverSendTime = ntpResponse.getServerSendTime();
                Long serverRecvTime = ntpResponse.getServerRecvTime();
                Long deviceSendTime = ntpResponse.getDeviceSendTime();
                TimeUtils.fillTime(serverSendTime, serverRecvTime, deviceSendTime, System.currentTimeMillis());
            }

            listener.onInitialized();
        });
    }
}
