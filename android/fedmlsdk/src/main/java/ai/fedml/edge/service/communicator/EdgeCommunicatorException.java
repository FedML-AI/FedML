package ai.fedml.edge.service.communicator;

public class EdgeCommunicatorException extends RuntimeException {
    public EdgeCommunicatorException(String message) {
        super(message);
    }

    public EdgeCommunicatorException(String message, Throwable cause) {
        super(message, cause);
    }
}
