import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "../../../")))
sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "../../../../FedML")))

from fedml_api.distributed.fedavg_cross_silo.message_define import MyMessage


class ClientStubObject:
    def __init__(
        self,
        client_id,
        os_name
    ):
        self.client_id = client_id
        self.os_name = os_name
        if os_name == MyMessage.MSG_CLIENT_OS_ANDROID or os_name == MyMessage.MSG_CLIENT_OS_IOS:
            self.is_mobile = True
        else:
            self.is_mobile = False
