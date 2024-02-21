from typing import List, Optional

from fedml.api.modules.utils import authenticate
from fedml.computing.scheduler.comm_utils.platform_utils import validate_platform
from fedml.computing.scheduler.scheduler_entry.constants import Constants
from fedml.computing.scheduler.scheduler_entry.run_manager import FedMLRunManager, FedMLRunLogModelList, \
    FedMLRunStartedModel, FedMLRunModelList, FeatureEntryPoint
from fedml.computing.scheduler.comm_utils.security_utils import get_api_key
from fedml.computing.scheduler.scheduler_entry.launch_manager import FedMLJobConfig
from fedml.api.constants import RunStatus


class RunLogResult(object):
    def __init__(self, run_status: RunStatus = None, total_log_lines: int = 0, total_log_pages: int = 0,
                 log_line_list: Optional[List[str]] = None, run_logs: FedMLRunLogModelList = None):
        self.run_status = run_status
        self.total_log_lines = total_log_lines
        self.total_log_pages = total_log_pages
        self.log_line_list = log_line_list
        self.run_logs = run_logs


def create(platform: str, job_config: FedMLJobConfig, device_server: str, device_edges: List[str],
           api_key: str, feature_entry_point: FeatureEntryPoint = None) -> FedMLRunStartedModel:
    _authenticate_and_validate_platform(api_key, platform)

    run_start_result = FedMLRunManager.get_instance().create_run(platform=platform, job_config=job_config,
                                                                 device_server=device_server, device_edges=device_edges,
                                                                 api_key=get_api_key(),
                                                                 feature_entry_point=feature_entry_point)

    return run_start_result


def create_on_cluster(platform: str, cluster: str, job_config: FedMLJobConfig, device_server: str,
                      device_edges: List[str], api_key: str,
                      feature_entry_point: FeatureEntryPoint = None) -> FedMLRunStartedModel:
    _authenticate_and_validate_platform(api_key, platform)

    run_start_result = FedMLRunManager.get_instance().create_run(platform=platform, job_config=job_config,
                                                                 device_server=device_server, device_edges=device_edges,
                                                                 api_key=get_api_key(), cluster=cluster,
                                                                 feature_entry_point=feature_entry_point)

    return run_start_result


def start(platform: str, create_run_result: FedMLRunStartedModel, device_server: str, device_edges: List[str],
          api_key: str, feature_entry_point: FeatureEntryPoint = None) -> FedMLRunStartedModel:
    _authenticate_and_validate_platform(api_key, platform)

    run_start_result = FedMLRunManager.get_instance().start_run(platform=platform, create_run_result=create_run_result,
                                                                device_server=device_server, device_edges=device_edges,
                                                                api_key=api_key,
                                                                feature_entry_point=feature_entry_point)

    return run_start_result


def stop(run_id: str, platform: str, api_key: str) -> bool:
    _authenticate_and_validate_platform(api_key, platform)

    is_stopped = FedMLRunManager.get_instance().stop_run(platform=platform, user_api_key=get_api_key(), run_id=run_id)
    return is_stopped


def list_run(run_name: str, run_id: str, platform: str, api_key: str) -> FedMLRunModelList:
    _authenticate_and_validate_platform(api_key, platform)

    if run_name is None and run_id is None:
        raise Exception("Please specify either run name or run id.")

    run_list_obj = FedMLRunManager.get_instance().list_run(platform=platform, project_name=None, run_name=run_name,
                                                           user_api_key=get_api_key(), run_id=run_id)
    return run_list_obj


def status(run_name: Optional[str], run_id: str, platform: str, api_key: str) -> (FedMLRunModelList, RunStatus):
    _authenticate_and_validate_platform(api_key, platform)

    run_status = None
    run_list_obj = list_run(run_name=run_name, run_id=run_id, platform=platform, api_key=api_key)

    if run_list_obj is not None:
        if len(run_list_obj.run_list) > 1:
            raise Exception("Found more than one runs for the specified run name or run id.")

        run_status = RunStatus.get_run_enum_from_str(run_list_obj.run_list[0].status)

    return run_list_obj, run_status


# input: run_id, page_num, page_size, need_all_logs, platform, api_key
# return RunLogResult(run_status, total_log_lines, total_log_pages, log_line_list, run_logs)
def logs(run_id: str, page_num: int, page_size: int, need_all_logs: bool, platform: str, api_key: str) -> RunLogResult:
    _authenticate_and_validate_platform(api_key, platform)

    if run_id is None:
        raise Exception("Please specify run id.")

    _, run_status = status(run_name=None, run_id=run_id, platform=platform, api_key=get_api_key())

    total_log_nums, total_log_pages, log_line_list, run_logs = 0, 0, list(), None

    if run_status is None:
        return RunLogResult()

    if not need_all_logs:
        run_logs = FedMLRunManager.get_instance().get_run_logs(run_id=run_id, page_num=page_num, page_size=page_size,
                                                               user_api_key=api_key)

        if run_logs is not None:
            total_log_pages, total_log_nums = run_logs.total_num, run_logs.total_pages
            _parse_logs(log_line_list, run_logs)

        return RunLogResult(run_status=run_status, total_log_lines=total_log_nums, total_log_pages=total_log_pages,
                            log_line_list=log_line_list, run_logs=run_logs)

    run_logs = FedMLRunManager.get_instance().get_run_logs(run_id=run_id, page_num=1,
                                                           page_size=Constants.RUN_LOG_PAGE_SIZE, user_api_key=api_key)

    if run_logs is None:
        return RunLogResult(run_status=run_status, total_log_lines=total_log_nums, total_log_pages=total_log_pages,
                            log_line_list=log_line_list, run_logs=run_logs)

    # Get all logs
    if len(run_logs.log_lines):
        _parse_logs(log_line_list, run_logs)

        for page_count in range(2, run_logs.total_pages + 1):
            run_logs = FedMLRunManager.get_instance().get_run_logs(run_id=run_id, page_num=page_count,
                                                                   page_size=Constants.RUN_LOG_PAGE_SIZE,
                                                                   user_api_key=api_key)
            _parse_logs(log_line_list, run_logs)

    return RunLogResult(run_status=run_status, total_log_lines=run_logs.total_num, total_log_pages=run_logs.total_pages,
                        log_line_list=log_line_list, run_logs=run_logs)


def _authenticate_and_validate_platform(api_key: str, platform: str) -> None:
    authenticate(api_key)
    validate_platform(platform)


def _parse_logs(log_line_list: List[str], run_logs: FedMLRunLogModelList) -> None:
    for log_line in run_logs.log_lines:
        log_line_list.append(log_line)
