
import copy
import logging
import multiprocessing
import time
import traceback
from multiprocessing import Process
from ..scheduler_core.account_manager import FedMLAccountManager
from .worker_agent import FedMLDeployWorkerAgent


class FedMLModelDeviceClientRunner:
    def __init__(self, args, current_device_id, os_name, is_from_docker, service_config, infer_host="127.0.0.1"):
        self.agent_process = None
        self.agent_runner = None
        self.agent_process_event = None
        self.args = copy.deepcopy(args)
        self.service_config = service_config
        self.unique_device_id = None
        self.current_device_id = current_device_id
        self.os_name = os_name
        self.is_from_docker = is_from_docker
        self.edge_id = None
        self.infer_host = infer_host
        self.redis_addr = "local"
        self.redis_port = "6379"
        self.redis_password = "fedml_default"

    def get_edge_id(self):
        return self.edge_id

    def start(self):
        self.agent_runner = FedMLModelDeviceClientRunner(self.args, self.current_device_id, self.os_name,
                                                         self.is_from_docker, self.service_config)
        self.agent_runner.infer_host = self.infer_host
        self.agent_runner.redis_addr = self.redis_addr
        self.agent_runner.redis_port = self.redis_port
        self.agent_runner.redis_password = self.redis_password
        if self.agent_process_event is None:
            self.agent_process_event = multiprocessing.Event()
        self.agent_process = Process(target=self.agent_runner.run_entry, args=(self.agent_process_event, self.args,))
        self.edge_id = self.bind_device()
        self.agent_process.start()

    def run_entry(self, process_event, in_args):
        # print(f"Model worker process id {os.getpid()}")

        self.agent_process_event = process_event

        worker_agent = FedMLDeployWorkerAgent()

        while not self.agent_process_event.is_set():
            try:
                try:
                    worker_agent.logout()
                except Exception as e:
                    pass

                worker_agent.login(
                    in_args.account_id, api_key=in_args.api_key, device_id=in_args.device_id,
                    os_name=in_args.os_name, role=FedMLAccountManager.ROLE_DEPLOY_WORKER_ON_PREM
                )
            except Exception as e:
                logging.info("Restart model device client: {}".format(traceback.format_exc()))
                pass
            finally:
                try:
                    worker_agent.logout()
                except Exception as e:
                    pass
                time.sleep(15)

        try:
            self.stop()
        except Exception as e:
            pass

    def check_runner_stop_event(self):
        if self.agent_process_event is not None and self.agent_process_event.is_set():
            logging.info("Received stopping event.")
            raise Exception("Runner stopped")

    def stop(self):
        FedMLDeployWorkerAgent.logout()

        if self.agent_process_event is not None:
            self.agent_process_event.set()

    def bind_device(self):
        # Login account
        login_result = FedMLAccountManager.get_instance().login(
            self.args.account_id, api_key=self.args.api_key, device_id=self.args.device_id,
            os_name=self.args.os_name, role=FedMLAccountManager.ROLE_DEPLOY_WORKER_ON_PREM
        )
        if login_result is not None:
            return login_result.edge_id
        else:
            return None
