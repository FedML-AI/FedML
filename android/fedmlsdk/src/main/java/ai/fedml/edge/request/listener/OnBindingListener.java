package ai.fedml.edge.request.listener;

import ai.fedml.edge.request.response.BindingResponse;

public interface OnBindingListener {
    void onDeviceBinding(BindingResponse.BindingData data);
}
