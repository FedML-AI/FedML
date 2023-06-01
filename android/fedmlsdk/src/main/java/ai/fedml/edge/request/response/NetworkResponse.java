package ai.fedml.edge.request.response;

public class NetworkResponse {
    int responseCode;
    String response;

    public NetworkResponse(int responseCode, String response) {
        this.responseCode = responseCode;
        this.response = response;
    }
}
