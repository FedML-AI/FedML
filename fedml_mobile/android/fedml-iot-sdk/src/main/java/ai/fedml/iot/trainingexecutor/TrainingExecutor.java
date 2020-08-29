package ai.fedml.iot.trainingexecutor;

public class TrainingExecutor {
    public static final String TRAINING_EXECUTOR_BASE_URL = "http://81.71.1.31";

    private static class LazyHolder {
        private static final TrainingExecutor sInstance = new TrainingExecutor();
    }

    public static TrainingExecutor getInstance() {
        return LazyHolder.sInstance;
    }
}
