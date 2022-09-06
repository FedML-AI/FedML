package ai.fedml.edge.request.response;

import lombok.Builder;
import lombok.Data;

@Data
@Builder
public class TokenResponse {
    private String code;
    private String message;
    private String data;
}
