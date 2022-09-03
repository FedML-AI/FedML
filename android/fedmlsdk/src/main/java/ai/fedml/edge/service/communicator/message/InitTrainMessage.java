package ai.fedml.edge.service.communicator.message;

import com.google.gson.annotations.SerializedName;

import java.util.List;

import lombok.AllArgsConstructor;
import lombok.Data;
import lombok.NoArgsConstructor;

@Data
@AllArgsConstructor
@NoArgsConstructor
public class InitTrainMessage {

    @SerializedName("edgeids")
    private List<Long> edgeIds;

    private Long id;

    private String name;
    @SerializedName("projectid")
    private String projectId;
    @SerializedName("run_config")
    private RunConfig runConfig;

    private Long runId;

    private Long starttime;

    private String timestamp;

    private String token;

    private List<Object> urls;

    private List<String> userids;

    @Data
    @AllArgsConstructor
    @NoArgsConstructor
    public static class RunConfig {


        private String configName;
        @SerializedName("data_config")
        private DataConfig dataConfig;
        @SerializedName("hyperparameters_config")
        private HyperParametersConfig hyperparametersConfig;
        @SerializedName("model_config")
        private ModelConfig modelConfig;
        @SerializedName("packages_config")
        private PackagesConfig packagesConfig;
    }

    @Data
    @AllArgsConstructor
    @NoArgsConstructor
    public static class DataConfig {


        private String privateLocalData;

        private String syntheticData;

        private String syntheticDataUrl;
    }

    @Data
    @AllArgsConstructor
    @NoArgsConstructor
    public static class HyperParametersConfig {


        private Double clientLearningRate;

        private Long clientNumPerRound;

        private String clientOptimizer;

        private Long commRound;

        private String communicationBackend;

        private String dataset;

        private Long localEpoch;

        private String partitionMethod;

        private Long trainBatchSize;
    }

    @Data
    @AllArgsConstructor
    @NoArgsConstructor
    public static class ModelConfig {
        private String modelName;
    }

    @Data
    @AllArgsConstructor
    @NoArgsConstructor
    public static class PackagesConfig {

        private String androidClient;

        private String androidClientUrl;

        private String androidClientVersion;

        private String linuxClient;

        private String linuxClientUrl;

        private String server;

        private String serverUrl;
    }
}
