import subprocess
from datetime import datetime, timezone, timedelta
import json
import yaml
import logging
import os
from typing import List, Tuple
from fedml.computing.scheduler.comm_utils.constants import SchedulerConstants
from fedml.computing.scheduler.comm_utils.container_utils import ContainerUtils
from fedml.computing.scheduler.comm_utils.hardware_utils import HardwareUtil
from fedml.computing.scheduler.comm_utils.scheduler_utils import SchedulerUtils
from fedml.computing.scheduler.slave.client_constants import ClientConstants

class CriClient:

    # all instances share the same container name
    shared_container_name = None
    shared_container_id = None

    def __init__(self):
        # self.runtime_endpoint = "unix:///run/containerd/containerd.sock"
        pass
    
    def _run_command(self, cmd):
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            return result.stdout.strip()
        except subprocess.CalledProcessError as e:
            logging.error(f"[CriClient] _run_command Error executing command: {e}")
            return None
    
    def get_container_name(self) -> str:
        # get container name from file and save it to class variable
        if CriClient.shared_container_name is None:
            current_model_dir = SchedulerUtils.get_current_model_dir()
            pod_name_file = os.path.join(current_model_dir, SchedulerConstants.K8S_POD_NAME_FILE)
            cmd = ["cat", pod_name_file]
            CriClient.shared_container_name = self._run_command(cmd).strip()
            logging.info(f"[CriClient] get_container_name: {CriClient.shared_container_name}")
        return CriClient.shared_container_name 
    
    def get_container_id(self):
        # use fixed shared container name because one pod only has one container-task
        if CriClient.shared_container_id is None:
            pod_name = self.get_container_name()
            # First get pod ID
            cmd = ["crictl", "pods", "--name", pod_name, "-q"]
            pod_id = self._run_command(cmd).strip()
            if pod_id:
                cmd = ["crictl", "ps", "-q", "--pod", pod_id, "--name", SchedulerConstants.K8S_POD_CONTAINER_TASK_NAME]
                CriClient.shared_container_id = self._run_command(cmd).strip()
                logging.info(f"[CriClient] get_container_id: {CriClient.shared_container_id}")
        return CriClient.shared_container_id
    
    def get_container_pid(self, container_id: str) -> int:
        """Get container PID using crictl inspect"""
        try:
            container_info = self.inspect_container(container_id)
            if not container_info:
                raise Exception("Failed to inspect container")
            return container_info['info']['pid']
        except Exception as e:
            logging.error(f"Failed to get container PID: {e}")
            raise
    
    def get_logs(self, container_id, since=None, follow=False, timestamps=False):
        """get container logs using crictl
        Args:
            container_id: container id
            since: show logs since a specific time
            follow: whether to follow the logs
            timestamps: whether to show timestamps
        """
        cmd = ["crictl", "logs"]
        
        if follow:
            cmd.append("-f")
        
        if since:
            cmd.extend(["--since", since])
            
        if timestamps:
            cmd.append("--timestamps")
            
        cmd.append(container_id)
        return self._run_command(cmd)
    
    def inspect_container(self, container_id):
        cmd = ["crictl", "inspect", container_id]
        output = self._run_command(cmd)
        if output:
            return json.loads(output)
        return None
    
    def get_network_stats(self, pid: int) -> Tuple[float, float]:
        """Get network statistics from container's network namespace"""
        try:
            # Read network statistics from the container's network namespace
            cmd = f"nsenter -t {pid} -n cat /proc/net/dev"
            result = subprocess.run(cmd.split(), capture_output=True, text=True, check=True)
            
            recv_bytes = 0
            sent_bytes = 0
            
            # Parse network interface statistics
            for line in result.stdout.split('\n')[2:]:  # Skip header lines
                if not line.strip():
                    continue
                parts = line.split()
                if len(parts) >= 10 and not parts[0].startswith('lo:'):  # Skip loopback
                    recv_bytes += int(parts[1])  # Received bytes
                    sent_bytes += int(parts[9])  # Transmitted bytes
            
            return (
                round(recv_bytes / (1024 * 1024), 1),  # Convert to MB
                round(sent_bytes / (1024 * 1024), 1)
            )
        except Exception as e:
            logging.error(f"Failed to get network stats: {e}")
            return 0.0, 0.0

    def get_blkio_stats(self, container_id: str) -> Tuple[float, float]:
        """Get block I/O statistics from cgroup"""
        try:
            # First try to find the cgroup path
            container_info = self.inspect_container(container_id)
            if not container_info:
                raise Exception("Failed to inspect container")
            
            # Get cgroup path from container info
            cgroup_path = container_info['info']['runtimeSpec']['linux']['cgroupsPath']
            
            # For containerd, the path might need to be adjusted
            if cgroup_path.startswith('system.slice'):
                cgroup_path = f"/sys/fs/cgroup/blkio/{cgroup_path}"
            
            read_bytes = 0
            write_bytes = 0
            
            # Read blkio statistics
            try:
                with open(f"{cgroup_path}/blkio.throttle.io_service_bytes", 'r') as f:
                    for line in f:
                        if "Read" in line:
                            read_bytes += int(line.split()[2])
                        elif "Write" in line:
                            write_bytes += int(line.split()[2])
            except FileNotFoundError:
                # Try alternative file for newer cgroup v2
                with open(f"{cgroup_path}/io.stat", 'r') as f:
                    for line in f:
                        if "rbytes" in line:
                            read_bytes += int(line.split("rbytes")[1].split()[0])
                        if "wbytes" in line:
                            write_bytes += int(line.split("wbytes")[1].split()[0])
            
            return (
                round(read_bytes / (1024 * 1024), 1),  # Convert to MB
                round(write_bytes / (1024 * 1024), 1)
            )
        except Exception as e:
            logging.error(f"Failed to get block I/O stats: {e}")
            return 0.0, 0.0

    def get_container_perf(self, container_id: str) -> ContainerUtils.ContainerMetrics:
        """
        Get container performance metrics using crictl stats
        
        Args:
            container_id: ID of the container to monitor
            
        Returns:
            ContainerMetrics object containing the performance data
            
        Raises:
            Exception: If unable to execute crictl or container not found
        """
        try:
            # Get container PID for network stats
            pid = self.get_container_pid(container_id)
            
            # Get basic stats from crictl
            cmd = f"crictl stats --id {container_id} --output json"
            result = subprocess.run(cmd.split(), capture_output=True, text=True, check=True)
            stats_data = json.loads(result.stdout)
            
            if not stats_data or 'stats' not in stats_data or not stats_data['stats']:
                raise Exception("No stats data returned from crictl")
            
            stats = stats_data['stats'][0]
            
            # Calculate CPU usage
            cpu_nano_cores = float(stats['cpu']['usageNanoCores']['value'])
            cpu_percent = round(cpu_nano_cores / 1e7, 2)  # Convert to percentage
            
            # Memory calculations
            mem_used_bytes = int(stats['memory']['usageBytes']['value'])
            mem_avail_bytes = int(stats['memory']['availableBytes']['value'])
            mem_used_mb = round(mem_used_bytes / (1024 * 1024), 1)
            mem_avail_mb = round(mem_avail_bytes / (1024 * 1024), 1)
            
            # # Storage calculations
            # storage_used_bytes = int(stats['writableLayer']['usedBytes']['value'])
            # storage_used_mb = round(storage_used_bytes / (1024 * 1024), 1)
            
            # # Inode usage
            # inodes_used = int(stats['writableLayer']['inodesUsed']['value'])
            
            # Get network stats
            network_recv_mb, network_sent_mb = self.get_network_stats(pid)
            
            # Get block I/O stats
            blk_read_mb, blk_write_mb = self.get_blkio_stats(container_id)
            
            # Convert timestamp from nanoseconds to readable format
            timestamp_ns = int(stats['cpu']['timestamp'])
            timestamp = datetime.fromtimestamp(timestamp_ns / 1e9).isoformat()

            # Calculate the gpu usage
            gpus_stat = self.generate_container_gpu_stats(container_name=container_id)

            # logging all stats
            logging.info(
                f"[CriClient] get_container_perf container_id: {container_id}, "
                f"cpu_percent: {cpu_percent}, mem_used_mb: {mem_used_mb}, "
                f"mem_avail_mb: {mem_avail_mb}, network_recv_mb: {network_recv_mb}, "
                f"network_sent_mb: {network_sent_mb}, blk_read_mb: {blk_read_mb}, "
                f"blk_write_mb: {blk_write_mb}, timestamp: {timestamp}, "
                f"gpus_stat: {gpus_stat}"
            )
            
            return ContainerUtils.ContainerMetrics(
                cpu_percent=cpu_percent,
                mem_used_megabytes=mem_used_mb,
                mem_avail_megabytes=mem_avail_mb,
                network_recv_megabytes=network_recv_mb,
                network_sent_megabytes=network_sent_mb,
                blk_read_megabytes=blk_read_mb,
                blk_write_megabytes=blk_write_mb,
                timestamp=timestamp,
                gpus_stat=gpus_stat
            )
            
        except subprocess.CalledProcessError as e:
            logging.error(f"Failed to execute crictl command: {e}")
            raise
        except Exception as e:
            logging.error(f"Error processing container stats: {e}")
            raise
    
    def generate_container_gpu_stats(self, container_name):
        gpu_ids = HardwareUtil.get_docker_gpu_ids_by_container_name(
            container_name=container_name,
            docker_client=None
        )
        gpu_stats = self.gpu_stats(gpu_ids)
        return gpu_stats

    @staticmethod
    def gpu_stats(gpu_ids: List[int]):
        utilz, memory, temp = None, None, None
        gpu_stats_map = {}  # gpu_id: int -> {"gpu_utilization", "gpu_memory_allocated", "gpu_temp"}
        gpu_ids = set(gpu_ids)
        try:
            for gpu in HardwareUtil.get_gpus():
                if gpu.id in gpu_ids:
                    gpu_stats_map[gpu.id] = {
                        "gpu_utilization": gpu.load * 100,
                        "gpu_memory_allocated": gpu.memoryUsed / gpu.memoryTotal * 100,
                        "gpu_temp": gpu.temperature,
                        # "gpu_power_usage": pynvml.nvmlDeviceGetPowerUsage(handle) / 1000,   # in watts
                        # "gpu_time_spent_accessing_memory": utilz.memory   # in ms
                    }
        except Exception as e:
            logging.error(f"Failed to get GPU stats: {e}")

        return gpu_stats_map

# example usage
if __name__ == "__main__":
    client = CriClient()
    
    # 获取容器ID
    container_id = client.get_container_id()
    if container_id:
        print(f"Container ID: {container_id}")
        
        # get and print gpu info
        gpu_info = client.get_gpu_info(container_id)
        if gpu_info:
            print(f"GPU Count: {gpu_info['count']}")
            print(f"GPU IDs: {', '.join(gpu_info['gpu_ids'])}")
        else:
            print("No GPU information found")
        
        # # get container info
        # info = client.inspect_container(container_id)
        # if info:
        #     print("Container info:", json.dumps(info, indent=2))
        
        # get recent 1 minute logs
        one_min_ago = (datetime.now(timezone.utc) - timedelta(minutes=1)).strftime("%Y-%m-%dT%H:%M:%SZ")
        logs = client.get_logs(container_id, since=one_min_ago)
        print("Recent logs:", logs)