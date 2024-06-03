import logging
import copy

from typing import List

from .device_model_cache import FedMLModelCache
from .device_model_msg_object import FedMLModelMsgObject
from .device_client_constants import ClientConstants


class FedMLDeviceReplicaController:
    def __init__(self, master_id, request_json: dict):
        """
        For each deployment, we have:
        master_id: unique id for the master device
        e_id: unique id (i.e. endpoint_id) for each deployment
        devices_avail_gpus = {device_id1: gpu_num, device_id2: gpu_num, ...}
        request_json: json from MLOps for this deployment
        total_gpu_num: total number of gpus will be used for this deployment
        gpu_per_replica: number of gpus required per replica
        min_replica_num: minimum number of replicas required
        max_replica_num: maximum number of replicas required
        endpoint_name: endpoint name
        model_name: model name
        target_replica_num: target replica number for each device
        target_replica_version: target replica version
        curr_replica_num: current replica number for each device
        intermediate_replica_num: intermediate replica number for each device
        total_replica_version_diff_num: total replica version difference number
        max_unavailable_rate: maximum unavailable rate
        curr_replica_updating_window: current replica updating window
        curr_replica_version: current replica version for each device
        intermediate_replica_version: intermediate replica version for each device
        """
        self.master_id = master_id
        self.request_json = request_json
        self.request_msg_obj = FedMLModelMsgObject("replica_controller", request_json)

        self.e_id = self.request_msg_obj.run_id
        self.devices_avail_gpus = self.request_msg_obj.gpu_topology
        self.total_gpu_num = self.calc_total_gpu_num()
        self.gpu_per_replica = self.request_msg_obj.gpu_per_replica
        self.min_replica_num = self.request_msg_obj.scale_min
        self.max_replica_num = self.request_msg_obj.scale_max
        self.endpoint_name = self.request_msg_obj.end_point_name
        self.model_name = self.request_msg_obj.model_name

        # Number control
        self.target_replica_num = self.init_id_replica_num()
        self.target_replica_ids = self.generate_replica_ids()

        self.curr_replica_num = self.get_curr_replica_num_state_frm_db()
        self.intermediate_replica_num = copy.deepcopy(self.curr_replica_num)

        # Version control
        self.target_replica_version = self.request_msg_obj.model_version
        self.max_unavailable_rate = self.request_msg_obj.max_unavailable_rate
        self.curr_replica_updating_window = {}

        self.start_version, self.curr_replica_version = self.get_curr_replica_version_frm_db()
        self.intermediate_replica_version = copy.deepcopy(self.curr_replica_version)

        self.total_replica_version_diff_num, self.total_replica_version_diff = self.diff_target_curr_replica_version()

        self.under_rollback = False

    def calc_total_gpu_num(self):
        total_gpu_num = 0
        for device_id, gpu_num in self.devices_avail_gpus.items():
            if type(gpu_num) is not int:
                logging.warning(f"The value in gpu_topology should be int, but got {type(gpu_num)}. Try to convert it.")
            total_gpu_num += int(gpu_num)
        return total_gpu_num

    def init_id_replica_num(self):
        """
        Initialize the target replica number for each device.
        id_replica_num[id] = avail_num // self.gpu_per_replica
        """
        id_replica_num = {}
        for id, avail_num in self.devices_avail_gpus.items():
            if type(avail_num) is not int:
                logging.warning(f"The value in gpu_topology should be int, "
                                f"but got {type(avail_num)}. Try to convert it.")
            avail_num = int(avail_num)

            if avail_num % self.gpu_per_replica != 0:
                raise ValueError("The number of gpus for each device should be divisible by gpu_per_replica")
            id_replica_num[str(id)] = avail_num // self.gpu_per_replica
        return id_replica_num

    def generate_replica_ids(self) -> List[str]:
        """
        [id1_replicaNo1, id2_replicaNo2, ...]
        replicaNo starts from 1
        """
        res = []
        for device_id, replica_num in self.target_replica_num.items():
            for i in range(replica_num):
                res.append(f"{device_id}_{i+1}")
        return res

    def diff_target_curr_replica_num(self):
        diff = self.diff_target_curr_replica_num_impl(self.target_replica_num, self.curr_replica_num)
        logging.info(
            f"[Replica Controller] [endpoint {self.e_id} ] <<<<< diff_target_curr_replica_num: {diff} >>>>>")
        return diff

    def diff_target_curr_replica_version(self):
        logging.info(f"[Replica Controller] [endpoint {self.e_id} ]"
                     f"target_replica_version: {self.target_replica_version}")
        logging.info(f"[Replica Controller] [endpoint {self.e_id} ]"
                     f"curr_replica_version: {self.curr_replica_version}")

        num_diff, diff = self.diff_target_curr_replica_version_impl(
            self.target_replica_version, self.curr_replica_version)

        logging.info(
            f"[Replica Controller] [endpoint {self.e_id} ] <<<<< diff_target_curr_replica_version: {diff} >>>>>")
        return num_diff, diff

    @staticmethod
    def diff_target_curr_replica_num_impl(target_replica_state, curr_replica_state):
        """
        Return the difference between target and current replica number.
        "op" could only be "add" or "remove".
        e.g.
        curr_replica_state = {id1: 1, id2: 1}
        target_replica_state = {id1: 2, id2: 2}

        return {id1: {"op": "add", "curr_num": 1, "target_num": 2}, id2: {"op": "add", "curr_num": 1, "target_num": 2}}
        """
        diff_target_curr_replica_num = {}
        assert target_replica_state is not None

        if curr_replica_state is None:
            curr_replica_state = {}
            for id, target_num in target_replica_state.items():
                diff_target_curr_replica_num[id] = {"op": "add", "curr_num": 0, "target_num": target_num}
            return diff_target_curr_replica_num

        for id, target_num in target_replica_state.items():
            if id not in curr_replica_state:
                # In one scale-out operation, the device may not be deployed yet.
                diff_target_curr_replica_num[id] = {"op": "add", "curr_num": 0, "target_num": target_num}
            elif target_num > curr_replica_state[id]:
                diff_target_curr_replica_num[id] = {"op": "add", "curr_num": curr_replica_state[id],
                                                    "target_num": target_num}
            elif target_num < curr_replica_state[id]:
                diff_target_curr_replica_num[id] = {"op": "remove", "curr_num": curr_replica_state[id],
                                                    "target_num": target_num}
            else:
                pass

        for id, curr_num in curr_replica_state.items():
            if id not in target_replica_state:
                diff_target_curr_replica_num[id] = {"op": "remove", "curr_num": curr_num, "target_num": 0}

        return diff_target_curr_replica_num

    @staticmethod
    def diff_target_curr_replica_version_impl(target_replica_version: str, curr_replica_version):
        """
        Return the number of difference, and difference between target and current replica version.
        "op" could only be "update".
        e.g.
        curr_replica_version = {
            "id1": {$replica_no: "v1", $replica_no: "v1"},
            "id2": {$replica_no: "v1", $replica_no: "v1"},
        }
        target_replica_version = "v2"   # Could be different for each device in the future.

        return {
            "id1": {
                $replica_no: {"op": "update", "new_version": "v2", "old_version": "v1"},
                $replica_no: {"op": "update", "new_version": "v2", "old_version": "v1"}
             },
            "id2": {
                $replica_no: {"op": "update", "new_version": "v2", "old_version": "v1"},
                $replica_no: {"op": "update", "new_version": "v2", "old_version": "v1"}
            }
        }

        Return None if curr_replica_version is None.(i.e. this model has not been deployed yet.)
        """
        if curr_replica_version is None:
            return 0, None

        diff_target_curr_replica_version = {}
        num_diff = 0
        for device_id, device_replicas_version in curr_replica_version.items():
            diff_target_curr_replica_version[device_id] = {}
            for replica_no, curr_version in device_replicas_version.items():
                if curr_version != target_replica_version:
                    num_diff += 1
                    diff_target_curr_replica_version[device_id][replica_no] = {
                        "op": "update",
                        "new_version": target_replica_version,
                        "old_version": curr_version
                    }
        if num_diff == 0:
            return 0, None

        return num_diff, diff_target_curr_replica_version

    def get_curr_replica_num_state_frm_db(self):
        """
        Sync the current replica number state from the database.
        Return the current replica number state.
        """
        res_frm_db = FedMLModelCache.get_instance().get_deployment_result_list(
            self.e_id, self.endpoint_name, self.model_name)

        curr_state = {}
        if res_frm_db is None or len(res_frm_db) == 0:
            # First time to get the replica number from the database
            for id, target_num in self.target_replica_num.items():
                curr_state[str(id)] = 0
        else:
            for result_item in res_frm_db:
                # Unpack the result_item
                result_device_id, _, result_payload = FedMLModelCache.get_instance().get_result_item_info(result_item)
                curr_state[str(result_device_id)] = curr_state.get(str(result_device_id), 0) + 1

        logging.info(f"[Replica Controller] [endpoint {self.e_id} ] curr_replica_state from db: {curr_state}")
        return curr_state

    def get_curr_replica_version_frm_db(self):
        """
        Sync the current replica version from the database.
        Return the current replica version.
        {
            "id1": {$replica_no: "v1", $replica_no: "v2"},
            "id2": {$replica_no: "v1", $replica_no: "v2"},
        }
        Return None if this model has not been deployed yet.
        """
        curr_versions = {}
        res_frm_db = FedMLModelCache.get_instance().get_deployment_result_list(
            self.e_id, self.endpoint_name, self.model_name)
        if res_frm_db is None or len(res_frm_db) == 0:
            return None, None
        else:
            version = None
            for result_item in res_frm_db:
                # Unpack the result_item
                result_device_id, replica_no, result_payload = (FedMLModelCache.get_instance().
                                                                get_result_item_info(result_item))
                if str(result_device_id) not in curr_versions:
                    curr_versions[str(result_device_id)] = {}
                curr_versions[str(result_device_id)][str(replica_no)] = result_payload["model_version"]

                if version is not None and version != result_payload["model_version"]:
                    logging.warning(f"Detected different model versions for the same endpoint in the same device.")
                version = result_payload["model_version"]

        return version, curr_versions

    def generate_diff_to_request_json(self):
        """
        Write the diff (curr <> target) to the self.request_json. e.g.
        {
            "replica_num_diff": {
                id1: {"op": "add", "curr_num": 1, "target_num": 2},
                id2: {"op": "add", "curr_num": 1, "target_num": 2},
                id3: {"op": "remove", "curr_num": 1, "target_num": 0}
            },
            "replica_version_diff": {
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
            "gpus_per_replica": 1,
        }
        """
        replica_num_diff_key = "replica_num_diff"
        gpu_per_replica_key = "gpus_per_replica"

        replica_num_diff = self.diff_target_curr_replica_num()
        self.request_json[replica_num_diff_key] = replica_num_diff

        self.request_json[gpu_per_replica_key] = self.gpu_per_replica
        return self.request_json

    def callback_update_curr_replica_num_state(self, changed_device_id, replica_no, op_type):
        """
        Callback function to update the current replica number.
        curr_state: {id1: 1, id2: 1}
        target_replica_state = {id1: 2, id2: 2}
        intermediate_state = {id1: 2, id2: 1}
        op_type: "add" or "remove"
        """
        if self.total_replica_version_diff_num != 0:
            # Should be viewed as updated, replica number will not be changed.
            return

        if str(changed_device_id) not in self.intermediate_replica_num:
            assert op_type == ClientConstants.MSG_MODELOPS_DEPLOYMENT_STATUS_DEPLOYED

            # Intermediate state is not initialized yet. Since it may derive from the database.
            self.intermediate_replica_num[str(changed_device_id)] = 0

        if op_type == ClientConstants.MSG_MODELOPS_DEPLOYMENT_STATUS_DEPLOYED:
            self.intermediate_replica_num[str(changed_device_id)] += 1
        elif op_type == ClientConstants.MSG_MODELOPS_DEPLOYMENT_STATUS_DELETED:
            self.intermediate_replica_num[str(changed_device_id)] -= 1

    def is_all_replica_num_reconciled(self):
        """
        Check if all the replicas are ready. Including the number and version.
        """
        logging.info(f"[Replica Controller] [Endpoint {self.e_id} ] "
                     f"intermediate_replica_num: {self.intermediate_replica_num}\n"
                     f"target_replica_num: {self.target_replica_num}")

        for id, replica_no in self.intermediate_replica_num.items():
            if id not in self.target_replica_num:   # Delete all replica in this device
                if replica_no != 0:
                    return False
                else:
                    continue
            if replica_no != self.target_replica_num[id]:
                return False

        for id, target_replica_num in self.target_replica_num.items():
            if id not in self.intermediate_replica_num or self.intermediate_replica_num[id] != target_replica_num:
                return False

        logging.info(f"[Replica Controller] [endpoint {self.e_id} ] Replicas are reconciled as expected.")
        return True

    def get_first_chunk_devices_replica_update(self):
        """
        Scroll update.
        Set the schema request json, which, will trans to subprocess (device_server_runner).
        The subprocess will send the init deployment msg to the worker device(s),
            then, the callback_deployment_result will handle the rest updating msg.

        e.g.
        {
            "replica_version_diff": {
                "id1": {
                    $replica_no: {"op": "update", "new_version": "v2", "old_version": "v1"},
                    $replica_no: {"op": "update", "new_version": "v2", "old_version": "v1"}
                 },
                "id2": {
                    $replica_no: {"op": "update", "new_version": "v2", "old_version": "v1"},
                    $replica_no: {"op": "update", "new_version": "v2", "old_version": "v1"}
                }
            },
        }

        Return None if there is no replica version difference.
        """
        if self.total_replica_version_diff_num == 0:
            return None

        window_size = max(1, int(self.total_replica_version_diff_num * self.max_unavailable_rate))

        first_chunk_devices_update = {}

        for device_id, device_replicas_version in self.total_replica_version_diff.items():
            for replica_no, diff in device_replicas_version.items():
                if len(first_chunk_devices_update) >= window_size:
                    break
                if device_id not in first_chunk_devices_update:
                    first_chunk_devices_update[device_id] = {}
                first_chunk_devices_update[device_id][replica_no] = diff

        return first_chunk_devices_update

    def init_update_updating_window(self, first_chunk_devices_update):
        """
        Initialize the current replica updating window.
        """
        self.curr_replica_updating_window = copy.deepcopy(first_chunk_devices_update)

    def callback_update_updating_window(self, device_id, replica_no):
        """
        Update the current replica updating window.
        """
        if str(device_id) not in self.curr_replica_updating_window:
            return

        if str(replica_no) not in self.curr_replica_updating_window[str(device_id)]:
            return

        # Remove the replica_no from the updating window
        del self.curr_replica_updating_window[str(device_id)][str(replica_no)]

        if len(self.curr_replica_updating_window[str(device_id)]) == 0:
            del self.curr_replica_updating_window[str(device_id)]

        # Change this replica's state in the global map
        self.intermediate_replica_version[str(device_id)][str(replica_no)] = self.target_replica_version

    def get_next_chunk_devices_replica(self):
        """
        If no need for updating, return None
        If the intermediate equal to target, return None
        If the current updating window is not empty, return None
        else, determine the next window, and send the request msg to the device -> replica handler.
        """
        if self.total_replica_version_diff_num == 0:
            return None

        if self.is_all_replica_version_reconciled():
            return None

        if len(self.curr_replica_updating_window) > 0:
            return None

        # Determine the next window
        window_size = max(1, int(self.total_replica_version_diff_num * self.max_unavailable_rate))

        next_chunk_devices_replicas_update = {}

        for id, device_replicas_version in self.intermediate_replica_version.items():
            for replica_no, version in device_replicas_version.items():
                if version != self.target_replica_version:
                    if id not in next_chunk_devices_replicas_update:
                        next_chunk_devices_replicas_update[id] = {}
                    next_chunk_devices_replicas_update[id][replica_no] = {
                        "op": "update",
                        "new_version": self.target_replica_version,
                        "old_version": version
                    }
                    if len(next_chunk_devices_replicas_update) >= window_size:
                        break

        return next_chunk_devices_replicas_update

    def is_all_replica_version_reconciled(self):
        """
        Check if all the replicas are ready. Including the number and version.
        """
        if self.total_replica_version_diff_num == 0:
            return True

        logging.info(f"[Replica Controller] [Endpoint {self.e_id} ] "
                     f"intermediate_replica_version: {self.intermediate_replica_version}\n"
                     f"target_replica_version: {self.target_replica_version}")

        for id, device_replicas_version in self.intermediate_replica_version.items():
            for replica_no, version in device_replicas_version.items():
                if version != self.target_replica_version:
                    return False

        logging.info(f"[Replica Controller] [endpoint {self.e_id} ] Replicas are reconciled as expected.")

        return True

    def init_first_update_device_replica_mapping(self):
        # Check if there is no replica version difference. return first_chunk_devices_update
        first_chunk_dict = self.get_first_chunk_devices_replica_update()
        if first_chunk_dict is None:
            return self.request_json

        # Update the updating window
        self.init_update_updating_window(first_chunk_dict)

        # Prepare and return the request json
        replica_version_diff_key = "replica_version_diff"
        self.request_json[replica_version_diff_key] = first_chunk_dict
        return self.request_json

    def rollback_get_replica_version_diff(self, device_id_trigger, replica_no_trigger):
        """
        for rollback existing replica that has been updated, get the replica version diff.
        rollback should be done at once, not rolling update.

        schema:
        {
            "replica_rollback_diff": {
                "id1": {
                    $replica_no: {"op": "rollback", "new_version": "v1", "old_version": "v2"},
                }
            },
        }
        """
        if self.start_version is None:
            return None

        devices_version_rollback = {}
        # TODO(Raphael): Consider the case that, multiple replicas in the chunk failed.
        for id, device_replicas_version in self.intermediate_replica_version.items():
            for replica_no, version in device_replicas_version.items():
                if version != self.start_version:
                    if id not in devices_version_rollback:
                        devices_version_rollback[id] = {}
                    devices_version_rollback[id][replica_no] = {
                        "op": "rollback",
                        "new_version": self.start_version,
                        "old_version": version
                    }
                # do not forget that the replica who triggered this failed callback, should also be rolled back.
                if id == device_id_trigger and replica_no == str(replica_no_trigger):
                    if id not in devices_version_rollback:
                        devices_version_rollback[id] = {}

                    devices_version_rollback[id][replica_no] = {
                        "op": "rollback",
                        "new_version": self.start_version,
                        "old_version": version
                    }
        self.under_rollback = True
        return devices_version_rollback

    def rollback_setback_target_replica_version(self):
        """
        Set back the target replica version.
        """
        self.target_replica_version = self.start_version

    def rollback_add_or_remove_replica(self, device_id, replica_no, op_type) -> dict:
        """
        During add or remove replica, in a specific step, the operation failed.

        if failed in delete replica, we should add the deleted replicas back.
        if failed in add replica, we should remove the added replicas.

        """
        reversed_diff = self.diff_target_curr_replica_num_impl(self.curr_replica_num, self.intermediate_replica_num)

        # Reverse the target replica number to the initial state.
        self.target_replica_num = copy.deepcopy(self.curr_replica_num)

        return reversed_diff
