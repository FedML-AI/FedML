package ai.fedml.edge;

public class FedEdgeManager {
    
    private static class LazyHolder {
        private final static FedEdgeApi S_FED_EDGE_API = new FedEdgeImpl();
    }

    public static FedEdgeApi getFedEdgeApi() {
        return LazyHolder.S_FED_EDGE_API;
    }
}
