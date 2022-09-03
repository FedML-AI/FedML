package ai.fedml.edge.request.listener;

import ai.fedml.edge.request.response.ConfigResponse;

public interface OnConfigListener {
    void onConfig(ConfigResponse.ConfigEntity entity);
}
