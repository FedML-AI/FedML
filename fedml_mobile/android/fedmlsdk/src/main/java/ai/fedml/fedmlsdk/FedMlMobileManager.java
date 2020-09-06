package ai.fedml.fedmlsdk;

public class FedMlMobileManager {
    private static class LazyHolder {
        private final static FedMlMobileApi sFedMlMobileApi = new FedMlMobileImpl();
    }

    public static FedMlMobileApi getFedMlMobileApi() {
        return LazyHolder.sFedMlMobileApi;
    }
}
