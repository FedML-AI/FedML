import uuid

from fedml_api.model.mobile.model_transfer import init_mnn_model, mnn_pytorch


def create_mobile_lenet_model():
    mnn_model_init_path = "/tmp/" + str(uuid.uuid4()) + ".mnn"
    init_mnn_model(mnn_model_init_path)
    return mnn_pytorch(mnn_model_init_path), mnn_model_init_path

