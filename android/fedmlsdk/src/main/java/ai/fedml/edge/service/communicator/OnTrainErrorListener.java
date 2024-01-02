package ai.fedml.edge.service.communicator;

public interface OnTrainErrorListener {
    void onTrainError(Throwable throwable);
}
