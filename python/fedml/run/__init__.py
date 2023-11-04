from fedml.computing.scheduler.scheduler_entry.run_manager import FedMLRunModelList
from fedml.run import internals


__all__ = [
    "stop",
    "list",
    "status",
    "logs",
]

def stop(run_id: str, platform: str = "falcon", api_key: str = None) -> bool:
    return internals.stop(run_id=run_id, platform=platform, api_key=api_key)


def list(run_name: str, run_id: str = None, platform: str = "falcon", api_key: str = None) -> FedMLRunModelList:
    return internals.list_run(run_name=run_name, run_id=run_id, platform=platform, api_key=api_key)


def status(run_name: str, run_id: str = None, platform: str = "falcon", api_key: str = None) -> (
FedMLRunModelList, str):
    return internals.status(run_name=run_name, run_id=run_id, platform=platform, api_key=api_key)


def logs(run_id: str, page_num: int = 1, page_size: int = 10, need_all_logs: bool = False, platform: str = "falcon",
         api_key: str = None) -> internals.RunLogResult:
    return internals.logs(run_id=run_id, page_num=page_num, page_size=page_size, need_all_logs=need_all_logs,
                          platform=platform, api_key=api_key)
