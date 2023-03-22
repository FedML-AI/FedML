package ai.fedml.edge.request.response;

import com.google.gson.annotations.SerializedName;

import lombok.AllArgsConstructor;
import lombok.Data;
import lombok.EqualsAndHashCode;
import lombok.NoArgsConstructor;
import lombok.ToString;

@EqualsAndHashCode(callSuper = true)
@Data
@NoArgsConstructor
@AllArgsConstructor
@ToString(callSuper = true)
public class ConfigResponse extends BaseResponse {

    @SerializedName("data")
    private ConfigEntity data;

    @Data
    @NoArgsConstructor
    @AllArgsConstructor
    public static class ConfigEntity {
        @SerializedName("mqtt_config")
        private MqttConfig mqttConfig;
        @SerializedName("s3_config")
        private S3Config s3Config;
    }

    @Data
    @NoArgsConstructor
    @AllArgsConstructor
    public static class MqttConfig {
        @SerializedName("BROKER_HOST")
        private String host;

        @SerializedName("BROKER_PORT")
        private int port;

        @SerializedName("MQTT_KEEPALIVE")
        private int keepAlive;

        @SerializedName("MQTT_USER")
        private String user;

        @SerializedName("MQTT_PWD")
        private String password;
    }

    @Data
    @NoArgsConstructor
    @AllArgsConstructor
    public static class S3Config {

        @SerializedName("CN_S3_AKI")
        private String ak;

        @SerializedName("CN_S3_SAK")
        private String sk;

        @SerializedName("CN_REGION_NAME")
        private String regionName;

        @SerializedName("BUCKET_NAME")
        private String bucket;
    }
}
