package ai.fedml.client;

import ai.fedml.client.entity.VersionUpdateResponse;
import retrofit2.Call;
import retrofit2.http.GET;

public interface VersionUpdater {
    @GET("/fedmlOpsServer/apk/latestVersion")
    Call<VersionUpdateResponse> getLatestVersion();
}
