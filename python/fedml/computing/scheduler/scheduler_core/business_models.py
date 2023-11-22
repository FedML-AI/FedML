import json


class LogsUploadModel(object):
    def __init__(self, logs_json):
        log_obj = json.loads(logs_json)
        self.run_id = log_obj.get("run_id")
        self.edge_id = log_obj.get("edge_id")
        self.log_list = log_obj.get("logs")
        self.err_list = log_obj.get("errors")
        self.log_source = log_obj.get("source")
        self.create_time = log_obj.get("create_time")
        self.update_time = log_obj.get("update_time")
        self.created_by = log_obj.get("created_by")
        self.updated_by = log_obj.get("updated_by")


'''
{
    "runId": "1723994069094502400",
    "edgeId": -1,
    "timeZone": "Asia/Shanghai",
    "pageSize": 100,
    "pageNum": 1,
    "type": 1
}
'''


class LogRequestModel(object):
    def __init__(self, request_json):
        self.run_id = request_json.get("runId")
        self.edgeId = request_json.get("edgeId")
        self.time_zone = request_json.get("timeZone")
        self.page_size = request_json.get("pageSize")
        self.page_num = request_json.get("pageNum")
        self.log_type = request_json.get("type")


'''
{
    "message": "Succeeded to process request",
    "code": "SUCCESS",
    "data": {
        "totalPages": 7,
        "pageNum": 1,
        "totalSize": 662,
        "logs": [
            "[FedML-Client @device-id-1723982303606214656] [Mon, 13 Nov 2023 17:19:52 +0800] [INFO]-----GPU Machine scheduling successful-----",
        ]
    }
}
'''


class LogResponseModel(object):
    def __init__(self, request_json=None):
        if request_json is None:
            request_json = {}
        self.run_id = request_json.get("run_id")
        self.edgeId = request_json.get("edgeId")
        self.total_pages = request_json.get("totalPages")
        self.total_size = request_json.get("totalSize")
        self.page_num = request_json.get("pageNum")
        self.logs = request_json.get("logs")

    def __init__(self, run_id, edge_id, total_pages, total_size, page_num, logs):
        self.run_id = run_id
        self.edgeId = edge_id
        self.total_pages = total_pages
        self.total_size = total_size
        self.page_num = page_num
        self.logs = logs


class MetricsModel(object):
    def __init__(self, metrics_json):
        pass
