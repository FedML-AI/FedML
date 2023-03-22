package ai.fedml.edge.request.parameter;

import com.google.gson.annotations.SerializedName;

import java.util.Arrays;
import java.util.Collections;
import java.util.List;

import lombok.Data;

@Data
public class ConfigReq {
    public static final String MQTT_CONFIG = "mqtt_config";
    public static final String S3_CONFIG = "s3_config";

    @SerializedName("config_name")
    private final List<String> configName = Collections.unmodifiableList(Arrays.asList(MQTT_CONFIG, S3_CONFIG));
}
