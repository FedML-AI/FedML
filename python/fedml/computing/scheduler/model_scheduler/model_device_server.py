import json
import logging
import multiprocessing
import os
import time
import traceback
from multiprocessing import Process

import click
from fedml.computing.scheduler.comm_utils.constants import SchedulerConstants

from fedml.computing.scheduler.model_scheduler import device_server_runner
from fedml.computing.scheduler.model_scheduler import device_server_constants


class FedMLModelDeviceServerRunner:
    def __init__(self, args, current_device_id, os_name, is_from_docker, service_config, infer_host="127.0.0.1"):
        self.agent_process = None
        self.agent_runner = None
        self.agent_process_event = None
        self.real_server_runner = None
        self.args = args
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
        self.agent_runner = None

    def get_edge_id(self):
        return self.edge_id

    def start(self):
        self.agent_runner = FedMLModelDeviceServerRunner(self.args, self.current_device_id, self.os_name,
                                                         self.is_from_docker, self.service_config)
        self.agent_runner.infer_host = self.infer_host
        self.agent_runner.redis_addr = self.redis_addr
        self.agent_runner.redis_port = self.redis_port
        self.agent_runner.redis_password = self.redis_password
        if self.agent_process_event is None:
            self.agent_process_event = multiprocessing.Event()
        self.agent_process = Process(target=self.agent_runner.run_entry, args=(self.agent_process_event,))
        self.edge_id = self.bind_device(init_params=False)
        self.agent_process.start()

    def run_entry(self, process_event):
        # print(f"Model master process id {os.getpid()}")

        self.agent_process_event = process_event

        while not self.agent_process_event.is_set():
            try:
                try:
                    if self.real_server_runner is not None:
                        self.real_server_runner.stop_agent()
                except Exception as e:
                    pass

                self.bind_device()

                self.start_agent()
            except Exception as e:
                logging.info("Restart model device server: {}".format(traceback.format_exc()))
                pass
            finally:
                try:
                    if self.real_server_runner is not None:
                        self.real_server_runner.stop_agent()
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
        if self.real_server_runner is not None:
            self.real_server_runner.stop_agent()

        if self.agent_process_event is not None:
            self.agent_process_event.set()

    def get_binding_unique_device_id(self, current_device_id, os_name, is_from_docker=False):
        role_str = "OnPremise"

        # Judge whether running from fedml docker hub
        is_from_fedml_docker_hub = False
        dock_loc_file = device_server_constants.ServerConstants.get_docker_location_file()
        if os.path.exists(dock_loc_file):
            is_from_fedml_docker_hub = True

        # Build unique device id
        is_from_k8s = device_server_constants.ServerConstants.is_running_on_k8s()
        if is_from_k8s:
            unique_device_id = current_device_id + "@" + os_name + ".MDA.K8S." + role_str + ".Master.Device"
        elif is_from_docker:
            unique_device_id = current_device_id + "@" + os_name + ".MDA.Docker." + role_str + ".Master.Device"
        else:
            unique_device_id = current_device_id + "@" + os_name + ".MDA." + role_str + ".Master.Device"

        if is_from_fedml_docker_hub:
            unique_device_id = current_device_id + "@" + os_name + ".MDA.DockerHub." + role_str + ".Master.Device"

        return unique_device_id

    def init_logs_param(self, edge_id):
        self.args.log_file_dir = device_server_constants.ServerConstants.get_log_file_dir()
        self.args.run_id = 0
        self.args.role = "server"
        self.args.edge_id = edge_id
        setattr(self.args, "using_mlops", True)
        setattr(self.args, "server_agent_id", edge_id)

    def bind_device(self, init_params=True):
        self.unique_device_id = self.get_binding_unique_device_id(self.current_device_id, self.os_name,
                                                                  self.is_from_docker)

        # Create client runner for communication with the FedML server.
        if self.real_server_runner is None:
            self.real_server_runner = device_server_runner.FedMLServerRunner(self.args)

        # Bind account id to the ModelOps platform.
        register_try_count = 0
        edge_id = -1
        user_name = None
        extra_url = None
        while register_try_count < 5:
            try:
                edge_id, user_name, extra_url = self.real_server_runner.bind_account_and_device_id(
                    self.service_config["ml_ops_config"]["EDGE_BINDING_URL"], self.args.account_id,
                    self.unique_device_id, self.os_name
                )
                if edge_id > 0:
                    self.real_server_runner.edge_id = edge_id
                    break
            except Exception as e:
                click.echo("{}\n{}".format(SchedulerConstants.ERR_MSG_BINDING_EXCEPTION_2, traceback.format_exc()))
                click.echo(SchedulerConstants.ERR_MSG_BINDING_EXIT_RETRYING)
                register_try_count += 1
                time.sleep(3)
                continue

        if edge_id <= 0:
            click.echo("")
            click.echo("Oops, you failed to login the FedML ModelOps platform.")
            click.echo("Please check whether your network is normal!")
            return
        self.edge_id = edge_id

        # Init runtime logs
        if init_params:
            setattr(self.args, "client_id", edge_id)
            self.real_server_runner.infer_host = self.infer_host
            self.real_server_runner.redis_addr = self.redis_addr
            self.real_server_runner.redis_port = self.redis_port
            self.real_server_runner.redis_password = self.redis_password
            self.init_logs_param(edge_id)
            self.real_server_runner.args = self.args
            self.real_server_runner.run_as_edge_server_and_agent = True
            self.real_server_runner.user_name = user_name

        return edge_id

    def start_agent(self):
        # Log arguments and binding results.
        # logging.info("login: unique_device_id = %s" % str(unique_device_id))
        # logging.info("login: edge_id = %s" % str(edge_id))
        self.real_server_runner.unique_device_id = self.unique_device_id
        device_server_constants.ServerConstants.save_runner_infos(self.current_device_id + "." + self.os_name,
                                                                  self.edge_id, run_id=0)

        # Setup MQTT connection for communication with the FedML server.
        self.real_server_runner.infer_host = self.infer_host
        self.real_server_runner.setup_agent_mqtt_connection(self.service_config)

        # Start mqtt looper
        self.real_server_runner.start_agent_mqtt_loop(should_exit_sys=False)
