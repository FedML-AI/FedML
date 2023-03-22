package ai.fedml.edge;

import org.junit.Test;

import ai.fedml.edge.service.component.AuthenticTool;
import ai.fedml.edge.request.AccessTokenService;
import ai.fedml.edge.request.response.TokenResponse;
import retrofit2.Call;
import retrofit2.Response;
import retrofit2.Retrofit;
import retrofit2.converter.gson.GsonConverterFactory;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

import java.io.IOException;

public class AuthToolUnitTest {
    @Test
    public void validateAuthCodeTest() {
        String secret = "4b292c09d3dc40f9a55af9385feea0fe";
        String content = "the data content";
        AuthenticTool authTool = new AuthenticTool(secret);
        String authCode = authTool.generateAuthCode(content);
        System.out.println(authCode);
        boolean isAuth = authTool.validateAuthCode(
                "038133C143137ED2C3CB92524149462F357DB1846B273BAC1DE405780AEA48F3", content);
        assertTrue(isAuth);
    }

    @Test
    public void checkAuthCodeTest() throws IOException {
        assertEquals(4, 2 + 2);
    }
}
