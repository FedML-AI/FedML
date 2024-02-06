import logging
import json


class FedMLModelObject(object):
    def __init__(self, model_json):
        self.id = model_json["id"]
        self.model_name = model_json["modelName"]
        self.model_type = model_json["modelType"]
        self.model_url = model_json["modelUrl"]
        self.commit_type = model_json["commitType"]
        self.description = model_json["description"]
        self.latest_model_version = model_json["latestModelVersion"]
        self.createTime = model_json["createTime"]
        self.update_time = model_json["updateTime"]
        self.owner = model_json["owner"]
        self.github_link = model_json["githubLink"]
        self.parameters = model_json["parameters"]
        self.model_version = model_json["modelVersion"]
        self.file_name = model_json["fileName"]
        self.is_init = model_json["isInit"]
        self.is_from_open = model_json["isFromOpen"]

    def show(self, prefix=""):
        logging.info(
            "{}model name: {}, model id: {}, model version: {}, model url: {}".format(
                prefix, self.model_name, self.id, self.model_version, self.model_url))


class FedMLModelList(object):
    def __init__(self, model_list_json):
        self.total_num = model_list_json["total"]
        self.total_page = model_list_json["totalPage"]
        self.page_num = model_list_json["pageNum"]
        self.page_size = model_list_json["pageSize"]
        model_list_data = model_list_json["data"]
        self.model_list = list()
        for model_obj_json in model_list_data:
            model_obj = FedMLModelObject(model_obj_json)
            self.model_list.append(model_obj)


class FedMLEndpointDetail(object):
    def __init__(self, endpoint_detail_json):
        self.endpoint_id = endpoint_detail_json.get("id")
        self.endpoint_name = endpoint_detail_json.get("endpointName")
        self.model_name = endpoint_detail_json.get("modelName")
        self.model_id = endpoint_detail_json.get("modelId")
        self.model_version = endpoint_detail_json.get("modelVersion")
        self.resource_type = endpoint_detail_json.get("resourceType")
        self.status = endpoint_detail_json.get("status")
        self.edge_id = endpoint_detail_json.get("edgeId")
        self.create_time = endpoint_detail_json.get("createTime")
        self.update_time = endpoint_detail_json.get("updateTime")
        self.replicas = endpoint_detail_json.get("replicas")
        self.online_replicas = endpoint_detail_json.get("onlineReplicas")
        self.min = endpoint_detail_json.get("min")
        self.max = endpoint_detail_json.get("max")
        self.inference_engine = endpoint_detail_json.get("inferenceEngine")
        self.model_task_id = endpoint_detail_json.get("modelTaskId")
        self.inference_url = endpoint_detail_json.get("requestUrl")
        input_json_str = endpoint_detail_json.get("inputJson", None)
        self.input_json = json.loads(input_json_str) if input_json_str is not None else None
