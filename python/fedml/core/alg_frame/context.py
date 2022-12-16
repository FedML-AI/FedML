from .params import Params
from ..common.singleton import Singleton

"""
key:
received_model_cid
sent_model_cid
ipfs_secret_key
test_data
acc_on_last_round
acc_on_aggregated_model

usage:
    Context().add("key", value)
    value = Context().get("key")
"""


class Context(Params, Singleton):
    # client related
    KEY_CLIENT_ID_LIST_IN_THIS_ROUND = "client_id_list_in_this_round"

    # data related
    KEY_TEST_DATA = "test_data"

    # model related
    KEY_CLIENT_MODEL_LIST = "client_model_list"
    KEY_RECEIVED_MODEL_CID = "received_model_cid"
    KEY_SENT_MODEL_CID = "sent_model_cid"

    # metric related
    KEY_METRICS_ON_LAST_ROUND = "metrics_on_last_round"
    KEY_METRICS_ON_AGGREGATED_MODEL = "metrics_on_aggregated_model"

    # communication/storage related
    KEY_IPFS_SECRET_KEY = "ipfs_secret_key"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
