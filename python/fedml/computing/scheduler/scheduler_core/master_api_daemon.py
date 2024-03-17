from fastapi import FastAPI, Request
from .log_manager import LogsManager
from .metrics_manager import MetricsManager
from ..comm_utils import  sys_utils
import os


class MasterApiDaemon(object):
    def __init__(self):
        pass

    def run(self):
        api = FastAPI()

        @api.post("/system/log/getLogs")
        @api.get("/system/log/getLogs")
        async def get_log(request: Request):
            input_json = await request.json()

            log_res_model = LogsManager.get_logs(input_json)
            response_dict = {
                "message": "Succeeded to process request", "code": "SUCCESS",
                "data": {
                    "totalPages": log_res_model.total_pages,
                    "pageNum": log_res_model.page_num,
                    "totalSize": log_res_model.total_size,
                    "logs": log_res_model.logs
                }
            }

            return response_dict

        @api.get("/fedml/api/v1/metrics")
        async def get_metrics(request: Request):
            input_json = await request.json()

            response_dict = MetricsManager.get_instance().get_metrics()

            return {"response": response_dict}

        @api.post("/fedml/api/v1/log")
        @api.post("/fedmlLogsServer/logs/update")
        async def update_log(request: Request):
            input_json = await request.json()
            response_text = ""

            LogsManager.save_logs(input_json)

            return {"response": str(response_text)}

        @api.get("/ready")
        async def ready():
            return {"status": "Success"}

        import uvicorn
        port = 30800
        if sys_utils.check_port("localhost", port):
            return

        cur_dir = os.path.dirname(__file__)
        fedml_base_dir = os.path.dirname(os.path.dirname(os.path.dirname(cur_dir)))
        uvicorn.run(api, host="0.0.0.0", port=port, reload=True, reload_delay=3, reload_dirs=fedml_base_dir)



