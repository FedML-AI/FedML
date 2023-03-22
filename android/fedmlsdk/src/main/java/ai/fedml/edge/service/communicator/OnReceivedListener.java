package ai.fedml.edge.service.communicator;

public interface OnReceivedListener {
    void onReceived(String topic, byte[] payload);
}
