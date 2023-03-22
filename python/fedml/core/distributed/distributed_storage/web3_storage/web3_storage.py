import os
import time
import pickle

from fedml.core.mlops.mlops_profiler_event import MLOpsProfilerEvent
from ...crypto import crypto_api
from .....core.alg_frame.context import Context

import httpx


class Web3Storage:
    def __init__(
            self, ipfs_config):
        self.ipfs_config = ipfs_config
        self.ipfs_upload_uri = ipfs_config.get("upload_uri", "https://api.web3.storage/upload")
        self.ipfs_download_uri = ipfs_config.get("download_uri", "ipfs.w3s.link2")

    def write_model(self, model):
        pickle_dump_start_time = time.time()
        model_pkl = pickle.dumps(model)
        secret_key = Context().get("ipfs_secret_key")
        if secret_key is not None and secret_key != "":
            secret_key = bytes(secret_key, 'UTF-8')
            model_pkl = crypto_api.encrypt(secret_key, model_pkl)
        MLOpsProfilerEvent.log_to_wandb(
            {"PickleDumpsTime": time.time() - pickle_dump_start_time}
        )
        ipfs_upload_start_time = time.time()
        result, model_url = self.storage_ipfs_upload_file(model_pkl)
        MLOpsProfilerEvent.log_to_wandb(
            {"Comm/send_delay": time.time() - ipfs_upload_start_time}
        )
        return model_url

    def read_model(self, message_key):
        message_handler_start_time = time.time()
        model_pkl, _ = self.storage_ipfs_download_file(message_key)
        secret_key = Context().get("ipfs_secret_key")
        if secret_key is not None and secret_key != "":
            secret_key = bytes(secret_key, 'UTF-8')
            model_pkl = crypto_api.decrypt(secret_key, model_pkl)
        MLOpsProfilerEvent.log_to_wandb(
            {"Comm/recieve_delay_s3": time.time() - message_handler_start_time}
        )
        unpickle_start_time = time.time()
        model = pickle.loads(model_pkl)
        MLOpsProfilerEvent.log_to_wandb(
            {"UnpickleTime": time.time() - unpickle_start_time}
        )
        return model

    def storage_ipfs_upload_file(self, file_obj):
        """Upload file to IPFS using web3.storage.

        Args:
            file: file-like object in byte mode.

        Returns:
            Response: (Successful, cid or error message)
        """
        token = self.ipfs_config["token"]
        res = httpx.post(
            self.ipfs_upload_uri,
            headers={"Authorization": "Bearer " + token},
            files={"file": file_obj},
            timeout=None,
        )
        content = res.json()
        file_cid = content.get("cid", None)
        if file_cid is None:
            return False, content.get("message")
        else:
            return True, file_cid

    def storage_ipfs_download_file(self, ipfs_cid, output_path=None):
        """Download file stored in IPFS.

        Args:
            cid (str): string describing location of the file.
            output_path (Optional[str]): if set file will be stored at this path.

        Returns:
            Response: (content, output_file_obj)
        """
        token = self.ipfs_config["token"]
        res = None
        for _ in range(3):
            try:
                res = httpx.get(f"https://{ipfs_cid}.{self.ipfs_download_uri}")
                if res.status_code == 200:
                    break
            except httpx.ReadTimeout:
                time.sleep(1)

        if res is None:
            return None, None

        content = res.content

        output_file_obj = None
        if output_path is not None:
            output_file_obj = open(output_path, "wb")
            output_file_obj.write(content)

        return content, output_file_obj
