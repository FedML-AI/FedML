package ai.example.websocket;

import org.junit.Test;

import java.io.IOException;


import ai.fedml.client.RetrofitManager;
import ai.fedml.edge.request.response.BaseResponse;
import ai.fedml.edge.request.parameter.BindingAccountReq;
import ai.fedml.edge.request.response.BindingResponse;
import ai.fedml.edge.request.response.UserInfoResponse;
import ai.fedml.client.entity.VersionUpdateResponse;
import retrofit2.Call;
import retrofit2.Response;

import static org.junit.Assert.*;

/**
 * Example local unit test, which will execute on the development machine (host).
 *
 * @see <a href="http://d.android.com/tools/testing">Testing documentation</a>
 */
public class ExampleUnitTest {
    @Test
    public void addition_isCorrect() {
        assertEquals(4, 2 + 2);
    }

    @Test
    public void VersionUpdateTest() throws IOException {
        Call<VersionUpdateResponse> call = RetrofitManager.getVersionUpdater().getLatestVersion();
        Response<VersionUpdateResponse> response = call.execute();
        System.out.println("bindingAccountTest onResponse: " + (response.body() != null ? response.body() : null));
    }
}