package ai.fedml.fedmlsdk.trainingexecutor;

import com.google.gson.annotations.SerializedName;

import lombok.Data;

@Data
public class ExecutorResponse {
    @SerializedName("errno")
    private String errno;
    @SerializedName("executorId")
    private String executorId;
    @SerializedName("executorTopic")
    private String executorTopic;
    @SerializedName("client_id")
    private String clientId;
    @SerializedName("training_task_args")
    private TrainingTaskParam trainingTaskArgs;

    @Data
    public static class TrainingTaskParam {
        @SerializedName("dataset")
        private String dataSet;
        @SerializedName("data_dir")
        private String dataDir;
        @SerializedName("partition_method")
        private String partitionMethod;
        @SerializedName("partition_alpha")
        private String partitionAlpha;
        @SerializedName("model")
        private String model;
        @SerializedName("client_num_per_round")
        private String clientNumPerRound;
        @SerializedName("comm_round")
        private String commRound;
        @SerializedName("epochs")
        private String epochs;
        @SerializedName("lr")
        private String lr;
        @SerializedName("wd")
        private String wd;
        @SerializedName("batch_size")
        private String batchSize;
        @SerializedName("frequency_of_the_test")
        private String frequencyOfTheTest;
        @SerializedName("is_mobile")
        private String isMobile;
    }
}
