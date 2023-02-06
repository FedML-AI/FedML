from fastapi import FastAPI, Request
from .client_data_interface import FedMLClientDataInterface


api = FastAPI()


@api.get('/')
def root():
    return {'message': 'FedML Client Local API Service!'}


@api.post('/fedml/api/v2/currentJobStatus')
async def get_current_job_status(request: Request):
    # Get json data
    input_json = await request.json()

    current_job = FedMLClientDataInterface.get_instance().get_current_job()
    response = {"jobId": current_job.job_id,
                "edgeId": current_job.edge_id,
                "startedTime": int(float(current_job.started_time)),
                "endedTime": int(float(current_job.ended_time)),
                "progress": current_job.progress, "ETA": int(current_job.eta),
                "failedTime": int(float(current_job.failed_time)),
                "errorCode": current_job.error_code,
                "msg": current_job.msg}

    return response


@api.post('/fedml/api/v2/historyJobStatus')
async def get_history_job_status(request: Request):
    # Get json data
    input_json = await request.json()

    responses = list()
    history_jobs = FedMLClientDataInterface.get_instance().get_history_jobs()
    for job_item in history_jobs.job_list:
        response = {"jobId": job_item.job_id,
                    "edgeId": job_item.edge_id,
                    "startedTime": int(float(job_item.started_time)),
                    "endedTime": int(float(job_item.ended_time)),
                    "failedTime": int(float(job_item.failed_time)),
                    "errorCode": job_item.error_code,
                    "msg": job_item.msg}
        responses.append(response)

    return responses
