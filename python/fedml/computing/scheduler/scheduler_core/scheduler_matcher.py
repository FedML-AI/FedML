import json

from fedml.computing.scheduler.comm_utils.constants import SchedulerConstants


class SchedulerMatcher:
    @staticmethod
    def parse_and_print_gpu_info_for_all_edges(active_edge_info_dict, should_print=True, show_gpu_list=False):
        gpu_count_for_all_edges = 0
        gpu_available_count_for_all_edges = 0
        for edge_id, edge_info in active_edge_info_dict.items():
            gpu_total_count = edge_info.get("gpuCoresTotal", 0)
            gpu_count_for_all_edges += gpu_total_count
            gpu_available_count = edge_info.get("gpuCoresAvailable", 0)
            gpu_available_count_for_all_edges += gpu_available_count
            gpu_available_ids = edge_info.get("gpu_available_ids", [])
            if show_gpu_list:
                gpu_list = edge_info.get("gpu_list", [])
                print(f"GPU List is as follows. {gpu_list}")
            if should_print:
                print(
                    f"GPUs on edge id {edge_id}: total count {gpu_total_count}, "
                    f"available count {gpu_available_count}, available ids {gpu_available_ids}"
                )
        if should_print:
            print(f"Total GPU count for all edges: {gpu_count_for_all_edges}")
            print(f"Available GPU count for all edges: {gpu_available_count_for_all_edges}")
        return gpu_count_for_all_edges, gpu_available_count_for_all_edges

    @staticmethod
    def get_master_node_info(edge_id_list, active_edge_info_dict):
        master_node_addr = None
        master_node_port = None
        for edge_id in edge_id_list:
            active_edge_info = active_edge_info_dict.get(str(edge_id), None)
            if active_edge_info is None:
                continue
            if master_node_addr is None:
                master_node_addr = active_edge_info.get("node_ip", None)
            if master_node_port is None:
                master_node_port = active_edge_info.get("node_port", None)
            if master_node_addr is not None and master_node_port is not None:
                break
        if len(edge_id_list) <= 1:
            master_node_addr = "localhost"
        if master_node_port is None:
            master_node_port = SchedulerConstants.JOB_MATCH_DEFAULT_MASTER_NODE_PORT

        return master_node_addr, master_node_port

    @staticmethod
    def generate_match_info_for_scheduler(
            edge_id, edge_id_list, master_node_addr, master_node_port, assigned_gpu_num_dict, assigned_gpu_ids_dict,
            model_master_device_id=None, model_slave_device_id=None, model_slave_device_id_list=None
    ):
        scheduler_info = dict()
        scheduler_info["master_node_addr"] = master_node_addr
        scheduler_info["master_node_port"] = master_node_port
        scheduler_info["num_nodes"] = len(edge_id_list)
        scheduler_info["matched_gpu_num"] = assigned_gpu_num_dict.get(str(edge_id), 0)
        scheduler_info["matched_gpu_ids"] = assigned_gpu_ids_dict.get(str(edge_id), list())
        scheduler_info["model_master_device_id"] = model_master_device_id if model_master_device_id is not None else ""
        scheduler_info["model_slave_device_id"] = model_slave_device_id if model_slave_device_id is not None else ""
        scheduler_info["model_slave_device_id_list"] = model_slave_device_id_list if model_slave_device_id_list is not None else []

        return scheduler_info

    @staticmethod
    def generate_new_edge_list_for_gpu_matching(assigned_gpu_num_dict):
        matched_edge_id_list = list()
        for edge_id, gpu_num in assigned_gpu_num_dict.items():
            if gpu_num <= 0:
                continue
            matched_edge_id_list.append(edge_id)

        return matched_edge_id_list

    @staticmethod
    def match_and_assign_gpu_resources_to_devices(request_gpu_num, edge_id_list, active_edge_info_dict, job_gpu_id_list=None):
        assigned_gpu_num_dict = dict()
        assigned_gpu_ids_dict = dict()
        if job_gpu_id_list is not None:
            job_gpu_id_list_json = json.loads(job_gpu_id_list)
            for edge_id, edge_info in active_edge_info_dict.items():
                gpu_count = job_gpu_id_list_json.get(f"gpu_{edge_id}")
                gpu_count = int(gpu_count) if gpu_count is not None else 1
                assigned_gpu_num_dict[str(edge_id)] = gpu_count
            return assigned_gpu_num_dict, assigned_gpu_ids_dict

        # Calculate total available gpu count
        total_available_gpu_count = 0
        for edge_id, edge_info in active_edge_info_dict.items():
            gpu_available_count = edge_info.get("gpuCoresAvailable", 0)
            total_available_gpu_count += gpu_available_count

        # Check if total available gpu count is less than request gpu num
        request_gpu_num = 0 if request_gpu_num is None or request_gpu_num < 0 else request_gpu_num
        if total_available_gpu_count < request_gpu_num:
            return None, None

        # First, allocate GPUs equally to each device.
        assigned_gpu_num = 0
        average_gpu_num_per_edge = int(request_gpu_num / len(edge_id_list))
        for edge_id, edge_info in active_edge_info_dict.items():
            gpu_available_count = edge_info.get("gpuCoresAvailable", 0)
            match_gpu_num = min(gpu_available_count, average_gpu_num_per_edge)
            assigned_gpu_num_dict[str(edge_id)] = match_gpu_num
            assigned_gpu_num += match_gpu_num

        # Add remaining GPUs to each device
        for edge_id, edge_info in active_edge_info_dict.items():
            gpu_available_count = edge_info.get("gpuCoresAvailable", 0)
            cur_gpu_num = assigned_gpu_num_dict[str(edge_id)]
            gpu_num_to_add = max(gpu_available_count - cur_gpu_num, 0)
            gpu_num_to_add = min(gpu_num_to_add, request_gpu_num - assigned_gpu_num)
            assigned_gpu_num += gpu_num_to_add
            assigned_gpu_num_dict[str(edge_id)] += gpu_num_to_add

        for edge_id, edge_info in active_edge_info_dict.items():
            cur_gpu_num = int(assigned_gpu_num_dict[str(edge_id)])
            gpu_available_ids = edge_info.get("gpu_available_ids", [])
            assigned_gpu_ids_dict[str(edge_id)] = gpu_available_ids[0:cur_gpu_num] if len(gpu_available_ids) > 0 else []

        return assigned_gpu_num_dict, assigned_gpu_ids_dict
