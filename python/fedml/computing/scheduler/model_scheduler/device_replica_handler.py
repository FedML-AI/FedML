import logging
from ..scheduler_core.compute_cache_manager import ComputeCacheManager
from ..comm_utils.container_utils import ContainerUtils
from ..comm_utils import security_utils
from .device_client_constants import ClientConstants
from .device_model_msg_object import FedMLModelMsgObject


class FedMLDeviceReplicaHandler:
    def __init__(self, worker_id, request_json: dict):
        """
        Handler on the worker to actually exec the reconciliation logic (Including add, remove, update).

        e_id: unique id (i.e. endpoint_id) for each deployment
        devices_avail_gpus = {device_id1: gpu_num, device_id2: gpu_num, ...}
        request_json: json from MLOps for this deployment
        total_gpu_num: total number of gpus will be used for this deployment
        gpu_per_replica: number of gpus required per replica
        """
        self.worker_id = worker_id
        self.request_json = request_json
        self.request_msg_obj = FedMLModelMsgObject("replica_handler", request_json)
        self.e_id = self.request_msg_obj.run_id
        self.gpu_per_replica = self.request_msg_obj.gpu_per_replica

        self.replica_num_diff = self.get_diff_replica_num_frm_request_json()
        self.is_rollback, self.replica_version_diff = self.get_diff_replica_version_frm_request_json()

        self.end_point_name = self.request_msg_obj.end_point_name
        self.inference_model_name = self.request_msg_obj.model_name
        self.model_version = self.request_msg_obj.model_version
        self.model_id = self.request_msg_obj.model_id

        self.device_avail_gpus = self.get_device_avail_gpus_frm_db()

    def get_device_avail_gpus_frm_db(self):
        """
        Get the available gpus from db.
        """
        available_gpu_ids = ComputeCacheManager.get_instance().get_gpu_cache().get_device_available_gpu_ids(
            self.worker_id)
        logging.info(f"[Replica Handler] [endpoint {self.e_id} ] [worker {self.worker_id}] "
                     f"All device_avail_gpus: {available_gpu_ids}")
        return available_gpu_ids

    def get_diff_replica_num_frm_request_json(self):
        """
        Read replica_diff passing by master's request json.
        Return:
        {
            id1_str: {"op": "add", "curr_num": 1, "target_num": 2},
            id2_str: {"op": "add", "curr_num": 1, "target_num": 2}
        }
        """
        if "replica_num_diff" in self.request_json and str(self.worker_id) in self.request_json["replica_num_diff"]:
            return self.request_json["replica_num_diff"][str(self.worker_id)]
        return None

    def get_diff_replica_version_frm_request_json(self):
        """
        Read replica_diff passing by master's request json.
        Return:
        {
            "id1": {
                $replica_no: {"op": "update", "new_version": "v2", "old_version": "v1"},
                $replica_no: {"op": "update", "new_version": "v2", "old_version": "v1"}
             },
            "id2": {
                $replica_no: {"op": "update", "new_version": "v2", "old_version": "v1"},
                $replica_no: {"op": "update", "new_version": "v2", "old_version": "v1"}
            }
        }
        """
        if ("replica_version_diff" in self.request_json and
                str(self.worker_id) in self.request_json["replica_version_diff"]):
            is_rollback = False
            for replica_no, diff in self.request_json["replica_version_diff"][str(self.worker_id)].items():
                if diff["op"] == "rollback":
                    is_rollback = True
                    break
            return is_rollback, self.request_json["replica_version_diff"][str(self.worker_id)]

        return None, None

    def reconcile_num_replica(self):
        """
        To solve the conflict between different reconciliation requests. The request & delete reqs should be
        executed in order and atomic (i.e. rollback).

        return (op, number of op)
        """
        if not self.replica_num_diff:
            logging.info(f"replica_num_diff is empty, will not reconcile.")
            return None, None, None

        if self.replica_num_diff["op"] not in ["add", "remove"]:
            raise ValueError(f"op should be add or remove. Got {self.replica_num_diff['op']}")

        prev_rank = self.replica_num_diff["curr_num"] - 1
        if self.replica_num_diff["op"] == "add":
            assert self.replica_num_diff["target_num"] > self.replica_num_diff["curr_num"]
            op, op_num = (self.replica_num_diff["op"],
                          self.replica_num_diff["target_num"] - self.replica_num_diff["curr_num"])
        else:
            assert self.replica_num_diff["target_num"] < self.replica_num_diff["curr_num"]
            op, op_num = (self.replica_num_diff["op"],
                          self.replica_num_diff["curr_num"] - self.replica_num_diff["target_num"])
        return prev_rank, op, op_num

    def remove_replica(self, rank):
        """
        Remove replica_num replicas from device_id.
        """
        running_model_name = ClientConstants.get_running_model_name(
            self.end_point_name, self.inference_model_name, self.model_version, self.e_id, self.model_id,
            self.worker_id)
        container_prefix = ("{}".format(ClientConstants.FEDML_DEFAULT_SERVER_CONTAINER_NAME_PREFIX) + "__" +
                            security_utils.get_content_hash(running_model_name))
        container_name = container_prefix + "__" + str(rank)
        logging.info(f"[Replica Handler] [Remove Replica] [Device {self.worker_id}] [Endpoint {self.e_id}]"
                     f" [Replica {rank}] [Container {container_name}]")
        ContainerUtils.get_instance().remove_container(container_name)

    def reconcile_replica_version(self):
        """
        Return a list of replica_rank to be updated.
        Giving {
                $replica_no: {"op": "update", "new_version": "v2", "old_version": "v1"},
                $replica_no: {"op": "update", "new_version": "v2", "old_version": "v1"}
             }
        or
        {
            $replica_no: {"op": "rollback", "new_version": "v2", "old_version": "v1"},
            $replica_no: {"op": "rollback", "new_version": "v2", "old_version": "v1"}
         }
        for all replicas, update the version. i.e. stop and remove the container, records in db, then start the new
        container, and report when the new container is ready.
        """
        if not self.replica_version_diff:
            logging.info(f"replica_version_diff is empty, will not reconcile.")
            return None, None

        replica_rank_to_update = []
        ret_op = "update" if not self.is_rollback else "rollback"

        for replica_no, diff in self.replica_version_diff.items():
            replica_rank_to_update.append(int(replica_no)-1)

        return replica_rank_to_update, ret_op
