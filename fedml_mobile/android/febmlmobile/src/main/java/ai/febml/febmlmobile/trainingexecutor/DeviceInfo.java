package ai.febml.febmlmobile.trainingexecutor;

import com.google.gson.annotations.SerializedName;

import lombok.Builder;
import lombok.Data;

@Builder
@Data
public class DeviceInfo {
    @SerializedName("deviceId")
    private String deviceId;
}
