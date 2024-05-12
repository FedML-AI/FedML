from fastapi import FastAPI, Request
from fedml.computing.scheduler.scheduler_core.log_manager import LogsManager
from fedml.computing.scheduler.scheduler_core.metrics_manager import MetricsManager
from fedml.computing.scheduler.comm_utils import sys_utils
from fedml.computing.scheduler.scheduler_core.compute_cache_manager import ComputeCacheManager
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

        @api.get("/get_job_status")
        async def get_job_status(job_id):
            ComputeCacheManager.get_instance().set_redis_params()
            job_status = ComputeCacheManager.get_instance().get_status_cache().get_job_status(job_id)
            return {"job_status": job_status}

        @api.get("/get_device_status_in_job")
        async def get_device_status_in_job(job_id, device_id):
            ComputeCacheManager.get_instance().set_redis_params()
            device_status_in_job = ComputeCacheManager.get_instance().get_status_cache().get_device_status_in_job(
                job_id, device_id)
            return {"device_status_in_job": device_status_in_job}

        import uvicorn
        port = 30800
        if sys_utils.check_port("localhost", port):
            return

        cur_dir = os.path.dirname(__file__)
        fedml_base_dir = os.path.dirname(os.path.dirname(os.path.dirname(cur_dir)))
        uvicorn.run(api, host="0.0.0.0", port=port)


