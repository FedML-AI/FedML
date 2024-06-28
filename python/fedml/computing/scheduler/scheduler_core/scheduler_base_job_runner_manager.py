
from abc import ABC, abstractmethod


class FedMLSchedulerBaseJobRunnerManager(ABC):

    def __init__(self):
        if not hasattr(self, "job_runners"):
            self.job_runners = dict()
        if not hasattr(self, "cloud_run_process_map"):
            self.cloud_run_process_map = dict()

    @abstractmethod
    def _generate_job_runner_instance(
            self, args, run_id=None, request_json=None, agent_config=None, edge_id=None
    ):
        return None

    def start_job_runner(
            self, run_id, request_json, args=None, edge_id=None, is_server_job=False,
            sender_message_queue=None, listener_message_queue=None, status_center_queue=None,
            should_start_cloud_server=False, use_local_process_as_cloud_server=False,
            cuda_visible_gpu_ids_str=None, process_name=None
    ):
        run_id_str = str(run_id)
        self.job_runners[run_id_str] = self._generate_job_runner_instance(
            args, run_id=run_id, request_json=request_json,
            agent_config=args.agent_config, edge_id=edge_id,
        )
        self.job_runners[run_id_str].start_runner_process(
            run_id, request_json, edge_id=edge_id,
            cuda_visible_gpu_ids_str=cuda_visible_gpu_ids_str,
            sender_message_queue=sender_message_queue,
            listener_message_queue=listener_message_queue,
            status_center_queue=status_center_queue,
            process_name=process_name
        )

    def stop_job_runner(self, run_id):
        run_id_str = str(run_id)
        if self.job_runners.get(run_id_str, None) is not None:
            self.job_runners[run_id_str].trigger_stop_event()

    def stop_all_job_runner(self):
        for run_id, job_runner in self.job_runners.items():
            job_runner.trigger_stop_event()

    def complete_job_runner(self, run_id):
        run_id_str = str(run_id)
        if self.job_runners.get(run_id_str, None) is not None:
            self.job_runners[run_id_str].trigger_completed_event()

    def put_run_edge_device_info_to_queue(self, run_id, edge_id, device_info):
        for job_run_id, job_runner in self.job_runners.items():
            job_runner.put_run_edge_device_info_to_queue(run_id, edge_id, device_info)

    def get_runner_process(self, run_id, is_cloud_server=False):
        run_id_str = str(run_id)

        if self.job_runners.get(run_id_str, None) is None:
            return None

        return self.job_runners[run_id_str].run_process

    def get_all_runner_pid_map(self):
        process_id_dict = dict()
        for run_id, runner in self.job_runners.items():
            if runner.run_process is not None:
                process_id_dict[str(run_id)] = runner.run_process.pid

        return process_id_dict
