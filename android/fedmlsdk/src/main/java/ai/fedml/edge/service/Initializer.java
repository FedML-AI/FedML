package ai.fedml.edge.service;

import ai.fedml.edge.request.RequestManager;
import ai.fedml.edge.request.response.ConfigResponse;
import androidx.annotation.NonNull;
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
            if (config.getS3Config() == null) {
                throw new RuntimeException("fetch s3 config failed!");
            }
            listener.onInitialized();
        });
    }
}
