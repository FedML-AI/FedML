package ai.fedml.edge.constants;

import ai.fedml.edge.BuildConfig;

public class UrlPath {

    private static final String BASE_API_SERVER_URL = BuildConfig.MLOPS_SVR;

    public static final String PATH_FETCH_CONFIG = BASE_API_SERVER_URL + "/fedmlOpsServer/configs/fetch";

    public static final String PATH_EDGE_BINDING = BASE_API_SERVER_URL + "/fedmlOpsServer/edges/binding";

    public static final String PATH_EDGE_UNBINDING = BASE_API_SERVER_URL + "/fedmlOpsServer/edges/unbound";

    public static final String PATH_USER_INFO = BASE_API_SERVER_URL + "/fedmlOpsServer/edges/device";

    public static final String PATH_LOG_UPLOAD = BASE_API_SERVER_URL + "/fedmlLogsServer/logs/update";
}
