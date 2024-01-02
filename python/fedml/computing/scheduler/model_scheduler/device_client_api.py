from fastapi import FastAPI, Request, Response, status
from fedml.computing.scheduler.model_scheduler.device_client_data_interface import FedMLClientDataInterface
from fedml.computing.scheduler.model_scheduler.device_http_inference_protocol import FedMLHttpInference


api = FastAPI()


@api.get('/')
def root():
    return {'message': 'FedML Model Client Local API Service!'}


@api.post('/fedml/api/v2/currentJobStatus')
async def get_current_job_status(request: Request):
    # Get json data
    input_json = await request.json()

    current_job = FedMLClientDataInterface.get_instance().get_current_job()
    if current_job is None:
        return {}
    response = {"jobId": current_job.job_id,
                "edgeId": current_job.edge_id,
                "startedTime": int(float(current_job.started_time)) if current_job.started_time != "" else 0,
                "endedTime": int(float(current_job.ended_time)) if current_job.ended_time != "" else 0,
                "progress": current_job.progress, "ETA": int(current_job.eta),
                "failedTime": int(float(current_job.failed_time))if current_job.ended_time != "" else 0,
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
                    "startedTime": int(float(job_item.started_time)) if job_item.started_time != "" else 0,
                    "endedTime": int(float(job_item.ended_time)) if job_item.ended_time != "" else 0,
                    "failedTime": int(float(job_item.failed_time))if job_item.ended_time != "" else 0,
                    "errorCode": job_item.error_code,
                    "msg": job_item.msg}
        responses.append(response)

    return responses


@api.post('/ready')
async def ready(request: Request, response: Response):
    input_json = await request.json()
    inference_url = input_json.get("inference_url", "0")
    inference_timeout = input_json.get("inference_timeout", None)

    response_ok = await FedMLHttpInference.is_inference_ready(inference_url, timeout=inference_timeout)
    if not response_ok:
        response.status_code = status.HTTP_404_NOT_FOUND
        return {'message': f'{inference_url} for inference is not ready.', 'status_code': response.status_code}

    return {'message': 'Http-proxy server for inference is ready.'}


@api.post('/api/v1/predict')
async def predict(request: Request, response: Response):
    # Get json data
    input_json = await request.json()
    endpoint_id = input_json.get("endpoint_id", None)
    if endpoint_id is None:
        return {}
    inference_url = input_json.get("inference_url", "0")
    inference_input_list = input_json.get("input", {})
    inference_output_list = input_json.get("output", [])
    inference_type = input_json.get("inference_type", "default")
    engine_type = input_json.get("engine_type", "default")
    inference_timeout = input_json.get("inference_timeout", None)

    response_ok, inference_response = await FedMLHttpInference.run_http_inference_with_curl_request(
        inference_url, inference_input_list, inference_output_list,
        inference_type=inference_type, engine_type=engine_type, timeout=inference_timeout)
    if not response_ok:
        response.status_code = status.HTTP_404_NOT_FOUND
        inference_response["status_code": response.status_code]

    return inference_response


