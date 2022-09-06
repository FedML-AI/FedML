package ai.fedml.edge.request;

import org.junit.Test;

import static org.junit.Assert.*;

public class RequestManagerTest {

    @Test
    public void getAccessToken() {
    }

    @Test
    public void fetchConfig() throws InterruptedException {
        System.out.println("==== fetchConfig ====");
        RequestManager.fetchConfig(data -> {
            System.out.println(data.toString());
        });
        Thread.sleep(5 * 1000L);
    }
}