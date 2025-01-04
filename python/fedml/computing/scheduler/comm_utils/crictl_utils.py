import subprocess
from datetime import datetime, timezone, timedelta
import json
import yaml
import logging
import os
from typing import List, Optional, Tuple
from fedml.computing.scheduler.comm_utils.constants import SchedulerConstants
from fedml.computing.scheduler.comm_utils.scheduler_utils import SchedulerUtils

class CriClient:

    # all instances share the same container name
    shared_container_name = None
    shared_container_id = None

    def __init__(self):
        # self.runtime_endpoint = "unix:///run/containerd/containerd.sock"
        pass
    
    def _run_command(self, cmd) -> Optional[str]:
        """Execute a command and return its output.
        Args:
            cmd: Command to execute (list or string)
        """
        try:
            if isinstance(cmd, str):
                cmd = cmd.split()
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            if result and result.stdout:
                return result.stdout.strip()
            return None
        except subprocess.CalledProcessError as e:
            logging.error(f"[CriClient] Command failed with exit code {e.returncode}: {e.cmd}")
            logging.error(f"Error output: {e.stderr}")
            return None
        except Exception as e:
            logging.error(f"[CriClient] Unexpected error executing command: {str(e)}")
            return None
    
    def get_container_name(self) -> str:
        # get container name from file and save it to class variable
        if CriClient.shared_container_name is None:
            current_model_dir = SchedulerUtils.get_current_model_dir()
            pod_name_file = os.path.join(current_model_dir, SchedulerConstants.K8S_POD_NAME_FILE)
            if not os.path.exists(pod_name_file):
                logging.error(f"[CriClient] Pod name file not found: {pod_name_file}")
                return None
            cmd = ["cat", pod_name_file]
            CriClient.shared_container_name = self._run_command(cmd)
            logging.info(f"[CriClient] get_container_name: {CriClient.shared_container_name}")
        return CriClient.shared_container_name 
    
    def get_container_id(self):
        # use fixed shared container name because one pod only has one container-task
        if CriClient.shared_container_id is None:
            pod_name = self.get_container_name()
            if not pod_name:
                logging.error("[CriClient] Failed to get container name")
                return None
            # First get pod ID
            cmd = ["crictl", "pods", "--name", pod_name, "-q"]
            pod_id = self._run_command(cmd)
            if not pod_id:
                logging.error("[CriClient] Failed to get pod ID")
                return None
            cmd = ["crictl", "ps", "-q", "--pod", pod_id, "--name", SchedulerConstants.K8S_POD_CONTAINER_TASK_NAME]
            container_id = self._run_command(cmd)
            if not container_id:
                logging.error("[CriClient] Failed to get container ID")
                return None
            CriClient.shared_container_id = container_id
            logging.info(f"[CriClient] get_container_id: {CriClient.shared_container_id}")
        return CriClient.shared_container_id
    
    def get_container_pid(self, container_id: str) -> int:
        """Get container PID using crictl inspect"""
        try:
            container_info = self.inspect_container(container_id)
            if not container_info:
                logging.error("[CriClient] Failed to inspect container")
                return None
            return container_info['info']['pid']
        except Exception as e:
            logging.error(f"Failed to get container PID: {e}")
            return None
        
    def get_logs(self, container_id, since=None, follow=False, timestamps=False):
        """get container logs using crictl
        Args:
            container_id: container id
            since: show logs since timestamp (e.g. '2013-01-02T13:23:37') 
                   or relative time (e.g. '42m' for 42 minutes)
            follow: whether to follow the logs
            timestamps: whether to show timestamps
        """
        cmd = ["crictl", "logs"]
        
        if follow:
            cmd.append("-f")
        
        if since:
            if isinstance(since, datetime):
                since = since.strftime("%Y-%m-%dT%H:%M:%SZ")
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
    
    def get_container_stats(self, container_id: str) -> dict:
        """Get container stats using crictl"""
        cmd = f"crictl stats --id {container_id} --output json"
        result = subprocess.run(cmd.split(), capture_output=True, text=True, check=True)
        stats_data = json.loads(result.stdout)
        
        if not stats_data or 'stats' not in stats_data or not stats_data['stats']:
            return None
            
        stats = stats_data['stats'][0]
        
        # Calculate CPU usage
        cpu_nano_cores = float(stats['cpu']['usageNanoCores']['value'])
        cpu_percent = round(cpu_nano_cores / 1e7, 2)  # Convert to percentage
        
        # Memory calculations
        mem_used_bytes = int(stats['memory']['usageBytes']['value'])
        mem_avail_bytes = int(stats['memory']['availableBytes']['value'])
        mem_used_mb = round(mem_used_bytes / (1024 * 1024), 1)
        mem_avail_mb = round(mem_avail_bytes / (1024 * 1024), 1)
        
        # Convert timestamp from nanoseconds to readable format
        timestamp_ns = int(stats['cpu']['timestamp'])
        timestamp = datetime.fromtimestamp(timestamp_ns / 1e9).isoformat()
        
        return {
            'cpu_percent': cpu_percent,
            'mem_used_mb': mem_used_mb,
            'mem_avail_mb': mem_avail_mb,
            'timestamp': timestamp
        }

    def get_network_stats(self, container_id: str) -> Tuple[float, float]:
        """Get network statistics from container's network namespace"""
        try:
            pid = self.get_container_pid(container_id)
            if not pid:
                logging.error("[CriClient] Failed to get container PID")
                return 0.0, 0.0
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
                logging.error("Failed to inspect container")
                return 0.0, 0.0
            
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