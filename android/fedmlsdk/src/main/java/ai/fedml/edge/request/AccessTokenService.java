package ai.fedml.edge.request;

import ai.fedml.edge.request.response.TokenResponse;
import retrofit2.Call;
import retrofit2.http.GET;
import retrofit2.http.Query;

public interface AccessTokenService {

    @GET("/fedmlOpsServer/groups/bygroupid")
    Call<TokenResponse> getAccessToken(@Query("groupId") String groupId, @Query("edgeId") String edgeId);
}
