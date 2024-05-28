import logging
import os
import time
import traceback
import fedml
from fedml.computing.scheduler.comm_utils import sys_utils
from .general_constants import GeneralConstants


class FedMLOtaUpgrade:
    LOCAL_RUNNER_INFO_DIR_NAME = 'runner_infos'
    STATUS_IDLE = "IDLE"

    def __init__(self, edge_id=None):
        self.edge_id = edge_id
        self.version = fedml.get_env_version()

    def ota_upgrade(self, payload, request_json, status_reporter=None,
                    is_master=False, is_deploy=False):
        run_id = request_json["runId"]
        force_ota = False
        ota_version = None

        try:
            run_config = request_json.get("run_config", None)
            parameters = run_config.get("parameters", None)
            common_args = parameters.get("common_args", None)
            force_ota = common_args.get("force_ota", False) if common_args is not None else False
            ota_version = common_args.get("ota_version", None) if common_args is not None else None
        except Exception as e:
            logging.error(
                f"Failed to get ota upgrade parameters with Exception {e}. Traceback: {traceback.format_exc()}")
            pass

        if force_ota and ota_version is not None:
            should_upgrade = True if ota_version != fedml.__version__ else False
            upgrade_version = ota_version
        else:
            try:
                fedml_is_latest_version, local_ver, remote_ver = sys_utils.check_fedml_is_latest_version(self.version)
            except Exception as e:
                logging.error(f"Failed to check fedml version with Exception {e}. Traceback: {traceback.format_exc()}")
                return

            should_upgrade = False if fedml_is_latest_version else True
            upgrade_version = remote_ver

        if should_upgrade:
            FedMLOtaUpgrade._save_upgrading_job(
                run_id, self.edge_id, payload, is_master=is_master, is_deploy=is_deploy
            )
            if status_reporter is not None:
                if is_master:
                    status_reporter.report_server_id_status(
                        run_id, GeneralConstants.MSG_MLOPS_SERVER_STATUS_UPGRADING, edge_id=self.edge_id,
                        server_id=self.edge_id, server_agent_id=self.edge_id)
                else:
                    status_reporter.report_client_id_status(
                        self.edge_id, GeneralConstants.MSG_MLOPS_CLIENT_STATUS_UPGRADING, run_id=run_id)

            logging.info(f"Upgrade to version {upgrade_version} ...")

            sys_utils.do_upgrade(self.version, upgrade_version)
            raise Exception("Restarting after upgraded...")

    @staticmethod
    def process_ota_upgrade_msg():
        os.system("pip install -U fedml")

    @staticmethod
    def _save_upgrading_job(run_id, edge_id, payload, is_master=False, is_deploy=False):
        if is_master and is_deploy:
            from ..model_scheduler.device_server_data_interface import FedMLServerDataInterface
            FedMLServerDataInterface.get_instance(). \
                save_started_job(run_id, edge_id, time.time(),
                                 GeneralConstants.MSG_MLOPS_CLIENT_STATUS_UPGRADING,
                                 GeneralConstants.MSG_MLOPS_CLIENT_STATUS_UPGRADING,
                                 payload)
        elif is_master and not is_deploy:
            from ..master.server_data_interface import FedMLServerDataInterface
            FedMLServerDataInterface.get_instance(). \
                save_started_job(run_id, edge_id, time.time(),
                                 GeneralConstants.MSG_MLOPS_CLIENT_STATUS_UPGRADING,
                                 GeneralConstants.MSG_MLOPS_CLIENT_STATUS_UPGRADING,
                                 payload)
        elif not is_master and is_deploy:
            from ..model_scheduler.device_client_data_interface import FedMLClientDataInterface
            FedMLClientDataInterface.get_instance(). \
                save_started_job(run_id, edge_id, time.time(),
                                 GeneralConstants.MSG_MLOPS_CLIENT_STATUS_UPGRADING,
                                 GeneralConstants.MSG_MLOPS_CLIENT_STATUS_UPGRADING,
                                 payload)
        elif not is_master and not is_deploy:
            from ..slave.client_data_interface import FedMLClientDataInterface
            FedMLClientDataInterface.get_instance(). \
                save_started_job(run_id, edge_id, time.time(),
                                 GeneralConstants.MSG_MLOPS_CLIENT_STATUS_UPGRADING,
                                 GeneralConstants.MSG_MLOPS_CLIENT_STATUS_UPGRADING,
                                 payload)
