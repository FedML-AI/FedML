import json
import os
import time
import pickle
import uuid

from fedml.core.mlops.mlops_profiler_event import MLOpsProfilerEvent
from ...crypto import crypto_api
from .....core.alg_frame.context import Context
from os.path import expanduser

import httpx
import shutil


class ThetaStorage:
    def __init__(
            self, thetasotre_config):
        self.ipfs_config = thetasotre_config
        self.store_home_dir = thetasotre_config.get("store_home_dir", "~/edge-store-playground")
        if str(self.store_home_dir).startswith("~"):
            home_dir = expanduser("~")
            new_store_dir = str(self.store_home_dir).replace('\\', os.sep).replace('/', os.sep)
            strip_dir = new_store_dir.lstrip('~').lstrip(os.sep)
            self.store_home_dir = os.path.join(home_dir, strip_dir)
        self.ipfs_upload_uri = thetasotre_config.get("upload_uri", "http://localhost:19888/rpc")
        self.ipfs_download_uri = thetasotre_config.get("download_uri", "http://localhost:19888/rpc")

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
            file_obj: file-like object in byte mode

        Returns:
            Response: (Successful, cid or error message)
        """
        # Request: upload a file
        # curl -X POST -H 'Content-Type: application/json' --data '{"jsonrpc":"2.0","method":"edgestore.PutFile","params":[{"path": "theta-edge-store-demos/demos/image/data/smiley_explorer.png"}],"id":1}' http://localhost:19888/rpc
        # Result
        # {
        #   "jsonrpc": "2.0",
        #   "id": 1,
        #   "result": {
        #       "key": "0xbc0383809da9fb98c5755e3fa4f19f4ebc7e34308ab321246e4bb54e548fad04",
        #       "relpath": "smiley_explorer.png",
        #       "success": true
        #   }
        # }
        home_dir = expanduser("~")
        file_path = os.path.join(home_dir, "thetastore")
        if not os.path.exists(file_path):
            os.makedirs(file_path)
        file_path = os.path.join(file_path, str(uuid.uuid4()))
        with open(file_path, "wb") as file_handle:
            file_handle.write(file_obj)

        request_data = {"jsonrpc":"2.0",
                "method":"edgestore.PutFile",
                "params":[{"path": file_path}],
                "id":1}
        res = httpx.post(
            self.ipfs_upload_uri,
            headers={"Content-Type": "application/json"},
            data=json.dumps(request_data),
            timeout=None,
        )
        content = res.json()
        result = content.get("result", None)
        if result is None:
            return False, "Failed to upload file(result is none)."
        else:
            success = result.get("success", False)
            if not success:
                return False, "Failed to upload file(success if false)."
            file_cid = result.get("key", None)
            if file_cid is None:
                return False, "Failed to upload file(key is none)."
            return True, file_cid

    def storage_ipfs_download_file(self, ipfs_cid, output_path=None):
        """Download file stored in IPFS.

        Args:
            cid (str): string describing location of the file.
            output_path (Optional[str]): if set file will be stored at this path.

        Returns:
            Response: (content, output_file_obj)
        """
        # Rquest: retrieve a file (the smiley_explorer.png file we uploaded earlier)
        # curl -X POST -H 'Content-Type: application/json' --data '{"jsonrpc":"2.0","method":"edgestore.GetFile","params":[{"key": "0xbc0383809da9fb98c5755e3fa4f19f4ebc7e34308ab321246e4bb54e548fad04"}],"id":1}' http://localhost:19888/rpc
        # Result
        # {
        #   "jsonrpc": "2.0",
        #   "id": 1,
        #   "result": {
        #       "path": "../data/edgestore/playground/single-node-network/node/storage/file_cache/0xbc0383809da9fb98c5755e3fa4f19f4ebc7e34308ab321246e4bb54e548fad04/smiley_explorer.png"
        #   }
        # }

        request_data = {"jsonrpc":"2.0",
                       "method":"edgestore.GetFile",
                       "params":[{"key": ipfs_cid}],
                       "id":1}
        res = httpx.post(
            self.ipfs_download_uri,
            headers={"Content-Type": "application/json"},
            data=json.dumps(request_data),
            timeout=None,
        )

        download_path = None
        content = res.json()
        result = content.get("result", None)
        if result is None:
            return False, "Failed to download file(result is none)."
        else:
            download_path = result.get("path", None)
            if download_path is None:
                return False, "Failed to download file(path is none)."
            else:
                download_path = os.path.join(self.store_home_dir, download_path)

        output_file_obj = None
        file_content = None
        try:
            if output_path is not None:
                shutil.copyfile(download_path, output_path)
                output_file_obj = open(output_path, "rb")
        except Exception as e:
            pass

        try:
            download_file_obj = open(download_path, "rb")
            file_content = download_file_obj.read()
        except Exception as e:
            pass

        return file_content, output_file_obj
