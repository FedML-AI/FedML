import base64
import json
import logging
import os
import time
from abc import ABC
from multiprocessing import Process
from .cloud_server_manager import FedMLCloudServerManager
from ..comm_utils.run_process_utils import RunProcessUtils
from ..scheduler_core.scheduler_base_job_runner_manager import FedMLSchedulerBaseJobRunnerManager


class FedMLBaseMasterJobRunnerManager(FedMLSchedulerBaseJobRunnerManager, ABC):
    def __init__(self):
        FedMLSchedulerBaseJobRunnerManager.__init__(self)

    # Override
    def start_job_runner(
            self, run_id, request_json, args=None, edge_id=None, is_server_job=False,
            sender_message_queue=None, listener_message_queue=None, status_center_queue=None,
            should_start_cloud_server=False, use_local_process_as_cloud_server=False,
            cuda_visible_gpu_ids_str=None
    ):
        if should_start_cloud_server:
            self._start_cloud_server(args, run_id, request_json, edge_id=edge_id,
                                     use_local_process_as_cloud_server=use_local_process_as_cloud_server)
            return

        run_id_str = str(run_id)
        self.job_runners[run_id_str] = self._generate_job_runner_instance(
            args, run_id=run_id, request_json=request_json,
            agent_config=args.agent_config, edge_id=edge_id,
        )
        self.job_runners[run_id_str].start_runner_process(
            run_id, request_json, edge_id=edge_id, is_server_job=is_server_job,
            sender_message_queue=sender_message_queue,
            listener_message_queue=listener_message_queue,
            status_center_queue=status_center_queue
        )

    def stop_job_runner(
            self, run_id, args=None, server_id=None, request_json=None,
            run_as_cloud_agent=False, run_as_cloud_server=False,
            use_local_process_as_cloud_server=False
    ):
        super().stop_job_runner(run_id)

        if run_as_cloud_agent or run_as_cloud_server:
            stopping_process = Process(
                target=FedMLCloudServerManager.stop_cloud_server,
                args=(run_id, server_id, args.agent_config))
            stopping_process.start()

            if run_as_cloud_server:
                time.sleep(1)
                RunProcessUtils.kill_process(os.getpid())

    def complete_job_runner(
            self, run_id, args=None, server_id=None, request_json=None,
            run_as_cloud_agent=False, run_as_cloud_server=False,
            use_local_process_as_cloud_server=False
    ):
        super().complete_job_runner(run_id)

        if run_as_cloud_agent or run_as_cloud_server:
            stopping_process = Process(
                target=FedMLCloudServerManager.stop_cloud_server,
                args=(run_id, server_id, args.agent_config))
            stopping_process.start()

            if run_as_cloud_server:
                time.sleep(1)
                RunProcessUtils.kill_process(os.getpid())

    def _start_cloud_server(
            self, args, run_id, request_json, edge_id=None,
            use_local_process_as_cloud_server=False
    ):
        run_id_str = str(run_id)
        cloud_server_mgr = FedMLCloudServerManager(
            args, run_id=run_id, edge_id=edge_id, request_json=request_json,
            agent_config=args.agent_config
        )
        if not use_local_process_as_cloud_server:
            self.cloud_run_process_map[run_id_str] = Process(target=cloud_server_mgr.start_cloud_server_process_entry)
            self.cloud_run_process_map[run_id_str].start()
        else:
            message_bytes = json.dumps(request_json).encode("ascii")
            base64_bytes = base64.b64encode(message_bytes)
            runner_cmd_encoded = base64_bytes.decode("ascii")
            cloud_device_id = request_json.get("cloudServerDeviceId", "0")

            logging.info("runner_cmd_encoded: {}".format(runner_cmd_encoded))

            self.cloud_run_process_map[run_id_str] = Process(
                target=cloud_server_mgr.start_local_cloud_server,
                args=(args.account_id, args.version, cloud_device_id, runner_cmd_encoded))
            self.cloud_run_process_map[run_id_str].start()
            time.sleep(1)

    def callback_run_logs(self, run_id, topic, payload):
        run_id_str = str(run_id)
        if self.job_runners.get(run_id_str, None) is not None:
            self.job_runners[run_id_str].callback_run_logs(topic, payload)

    def callback_run_metrics(self, run_id, topic, payload):
        run_id_str = str(run_id)
        if self.job_runners.get(run_id_str, None) is not None:
            self.job_runners[run_id_str].callback_run_metrics(topic, payload)
