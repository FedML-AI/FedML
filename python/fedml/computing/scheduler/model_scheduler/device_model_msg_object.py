
import json


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
        self.scale_min = self.model_config["instance_scale_min"]
        self.scale_max = self.model_config["instance_scale_max"]
        self.inference_engine = self.model_config.get("inference_engine", 0)
        self.inference_end_point_id = self.run_id

        self. request_json["run_id"] = self.run_id

    def show(self, prefix=""):
        print("{}end point id: {}, model name: {}, model id: {},"
              " model version: {}, model url: {}".format(prefix,
                                                         self.inference_end_point_id,
                                                         self.model_name,
                                                         self.id,
                                                         self.model_version,
                                                         self.model_url))
