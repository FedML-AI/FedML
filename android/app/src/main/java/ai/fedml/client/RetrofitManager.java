package ai.fedml.client;

import java.util.concurrent.TimeUnit;

import ai.fedml.edge.BuildConfig;
import ai.fedml.edge.request.UserManagerService;
import okhttp3.OkHttpClient;
import okhttp3.logging.HttpLoggingInterceptor;
import retrofit2.Retrofit;
import retrofit2.converter.gson.GsonConverterFactory;

public final class RetrofitManager {
    private static final String BASE_API_SERVER_URL = BuildConfig.MLOPS_SVR;
    private static Retrofit retrofit;
    private static UserManagerService userManagerService;
    private static VersionUpdater versionUpdater;

    private static Retrofit retrofit() {
        if (retrofit == null) {
            HttpLoggingInterceptor loggingInterceptor = new HttpLoggingInterceptor()
                    .setLevel(HttpLoggingInterceptor.Level.BASIC);

            OkHttpClient okHttpClient = new OkHttpClient.Builder()
                    .writeTimeout(30_1000, TimeUnit.MILLISECONDS)
                    .readTimeout(20_1000, TimeUnit.MILLISECONDS)
                    .connectTimeout(15_1000, TimeUnit.MILLISECONDS)
                    .addInterceptor(loggingInterceptor)
                    .build();

            retrofit = new Retrofit.Builder()
                    .baseUrl(BASE_API_SERVER_URL)
                    .addConverterFactory(GsonConverterFactory.create())
                    .client(okHttpClient)
                    .build();
        }
        return retrofit;
    }

    public static UserManagerService getOpsUserManager() {
        if (userManagerService == null) {
            userManagerService = retrofit().create(UserManagerService.class);
        }
        return userManagerService;
    }

    public static VersionUpdater getVersionUpdater() {
        if (versionUpdater == null) {
            versionUpdater = retrofit().create(VersionUpdater.class);
        }
        return versionUpdater;
    }
}
