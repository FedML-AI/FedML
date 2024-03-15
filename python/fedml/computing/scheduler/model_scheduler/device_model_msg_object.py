import json
import logging


class FedMLModelMsgObject(object):
    def __init__(self, topic, payload):
        """{
            "timestamp":1671440005119,
            "end_point_id":4325,
            "token":"FCpWU",
            "state":"STARTING",
            "user_id":"105",
            "user_name":"alex.liang2",
            "device_ids":[
                693
            ],
            "device_objs":[
                {
                    "device_id":"0xT3630FW2YM@MacOS.Edge.Device",
                    "os_type":"MacOS",
                    "id":693,
                    "ip":"1.1.1.1",
                    "memory":1024,
                    "cpu":"1.7",
                    "gpu":"Nvidia",
                    "extra_infos":{
                    }
                }
            ],
            "model_config":{
                "model_name":"image-model",
                "model_id":111,
                "model_storage_url":"https://fedml.s3.us-west-1.amazonaws.com/1666239314792client-package.zip",
                "model_version":"v1",
                "inference_engine":"onnx"
            }
        }"""

        # get deployment params
        if isinstance(payload, dict):
            request_json = payload
        else:
            request_json = json.loads(payload)
        self.msg_topic = topic
        self.request_json = request_json
        self.run_id = request_json["end_point_id"]
        self.end_point_name = request_json["end_point_name"]
        self.token = request_json["token"]
        self.user_id = request_json["user_id"]
        self.user_name = request_json["user_name"]
        self.device_ids = request_json["device_ids"]
        self.device_objs = request_json["device_objs"]

        self.model_config = request_json["model_config"]
        self.model_name = self.model_config["model_name"]
        self.model_id = self.model_config["model_id"]
        self.model_version = self.model_config["model_version"]
        self.model_storage_url = self.model_config["model_storage_url"]
        self.scale_min = self.model_config.get("instance_scale_min", 0)
        self.scale_max = self.model_config.get("instance_scale_max", 0)
        self.inference_engine = self.model_config.get("inference_engine", 0)
        self.inference_end_point_id = self.run_id

        self.request_json["run_id"] = self.run_id

        self.gpu_topology = self.get_devices_avail_gpus()
        self.gpu_per_replica = self.get_gpu_per_replica()

        self.max_unavailable_rate = self.model_config.get("max_unavailable_rate", 0.1)

    def get_devices_avail_gpus(self):
        """
        {
            "gpu_topology": {"id1": 1, "id2": 1}    # Here the 1 means gpu card, not replica
        }
        """
        # [Test1] using self.request_json["parameters"]["gpu_topology"]
        # logging.info(f"[Replica Controller] [endpoint {self.run_id} ] devices_avail_gpus:"
        #              f" {self.request_json['parameters']['gpu_topology']}")
        # res = self.request_json["parameters"]["gpu_topology"]

        # [Test2] Using self.scale_min
        # res = {}
        # for id in self.request_json["device_ids"]:
        #     if str(id) == str(self.device_ids[0]):
        #         continue
        #     res[id] = int(self.scale_min)
        # return res

        # [Prod] Using self.request_json["gpu_topology"]
        if "gpu_topology" not in self.request_json:
            logging.warning("gpu_topology not found in request_json, using scale_min instead")
            res = {}
            for id in self.request_json["device_ids"]:
                if str(id) == str(self.device_ids[0]):
                    continue
                res[id] = int(self.scale_min)
            return res

        logging.info(f"[Replica Controller] [endpoint {self.run_id}] "
                     f"devices_avail_gpus: {self.request_json['gpu_topology']}")

        return self.request_json["gpu_topology"]

    def get_gpu_per_replica(self) -> int:
        """
        Read gpu_per_replica from user's config yaml file. Default 1.
        """
        if "gpu_per_replica" in self.request_json:
            return int(self.request_json["gpu_per_replica"])
        return 1

    def show(self, prefix=""):
        logging.info(f"{prefix} [FedMLModelMsgObject] [run_id {self.run_id}] [end_point_name {self.end_point_name}]")
