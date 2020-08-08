package ai.fedml.iot.http.strategy;

import retrofit2.Call;
import retrofit2.http.Field;
import retrofit2.http.FormUrlEncoded;
import retrofit2.http.GET;
import retrofit2.http.PUT;
import retrofit2.http.Path;

public interface TrajectoryStrategyApi {
    @GET("trajectorystrategy/{IOT_UUID}")
    Call<ResultTrajectoryStrategy<TrajectoryStrategy>> getTrajectoryStrategy(@Path("IOT_UUID") String IOT_UUID);

    @FormUrlEncoded
    @PUT("trajectorystrategy/{IOT_UUID}")
    Call<ResultTrajectoryStrategy<TrajectoryStrategy>> updateTrajectoryStrategyByID(@Path("IOT_UUID") String IOT_UUID, @Field("strategy") String jsonStrategy);
}
