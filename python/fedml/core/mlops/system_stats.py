import os

from wandb.sdk.internal.settings_static import SettingsStatic
from .stats_impl import WandbSystemStats


class SysStats:
    def __init__(self, process_id=None):
        settings = SettingsStatic(d={"_stats_pid": os.getpid() if process_id is None else process_id})
        self.sys_stats_impl = WandbSystemStats(settings=settings, interface=None)
        self.gpu_time_spent_accessing_memory = 0.0
        self.gpu_power_usage = 0.0
        self.gpu_temp = 0.0
        self.gpu_memory_allocated = 0.0
        self.gpu_utilization = 0.0
        self.network_traffic = 0.0
        self.disk_utilization = 0.0
        self.process_cpu_threads_in_use = 0
        self.process_memory_available = 0.0
        self.process_memory_in_use = 0.0
        self.process_memory_in_use_size = 0.0
        self.system_memory_utilization = 0.0
        self.cpu_utilization = 0.0

    def produce_info(self):
        stats = self.sys_stats_impl.stats()

        self.cpu_utilization = stats.get("cpu", 0.0)
        self.system_memory_utilization = stats.get("memory", 0.0)
        self.process_memory_in_use_size = stats.get("proc.memory.percent", 0.0)
        self.process_memory_in_use = stats.get("proc.memory.rssMB", 0.0)
        self.process_memory_available = stats.get("proc.memory.availableMB", 0.0)
        self.process_cpu_threads_in_use = stats.get("proc.cpu.threads", 0.0)
        self.disk_utilization = stats.get("disk", 0.0)
        network = stats.get("network", None)
        if network is not None:
            self.network_traffic = network.get("sent", 0.0) + network.get("recv", 0.0)

        for stat_key, stat_value in stats.items():
            if str(stat_key).find("gpu.0.gpu") != -1:
                if self.sys_stats_impl.gpu_count == 0:
                    self.sys_stats_impl.gpu_count = 1

        gpu_mem_used = 0.0
        gpu_usage_total = 0.0
        gpu_mem_allocated = 0.0
        gpu_temperature_total = 0.0
        gpu_power_usage_total = 0.0

        if self.sys_stats_impl.gpu_count >= 1:
            for i in range(self.sys_stats_impl.gpu_count):
                gpu_mem_used += stats.get("gpu.{}.{}".format(i, "memory"), 0.0)
                gpu_usage_total += stats.get("gpu.{}.{}".format(i, "gpu"), 0.0)
                gpu_mem_allocated += stats.get("gpu.{}.{}".format(i, "memoryAllocated"), 0.0)
                gpu_temperature_total += stats.get("gpu.{}.{}".format(i, "temp"), 0.0)
                gpu_power_usage_total += stats.get("gpu.{}.{}".format(i, "powerPercent"), 0.0)

            self.gpu_utilization = round(
                gpu_usage_total / self.sys_stats_impl.gpu_count, 2
            )
            self.gpu_memory_allocated = round(
                gpu_mem_allocated / self.sys_stats_impl.gpu_count
            )
            self.gpu_temp = round(gpu_temperature_total / self.sys_stats_impl.gpu_count)
            self.gpu_power_usage = round(
                gpu_power_usage_total / self.sys_stats_impl.gpu_count
            )
            self.gpu_time_spent_accessing_memory = round(
                gpu_mem_used / self.sys_stats_impl.gpu_count
            )

    def get_cpu_utilization(self):
        return self.cpu_utilization

    def get_system_memory_utilization(self):
        return self.system_memory_utilization

    def get_process_memory_in_use(self):
        return self.process_memory_in_use

    def get_process_memory_in_use_size(self):
        return self.process_memory_in_use_size

    def get_process_memory_available(self):
        return self.process_memory_available

    def get_process_cpu_threads_in_use(self):
        return self.process_cpu_threads_in_use

    def get_disk_utilization(self):
        return self.disk_utilization

    def get_network_traffic(self):
        return self.network_traffic

    def get_gpu_utilization(self):
        return self.gpu_utilization

    def get_gpu_temp(self):
        return self.gpu_temp

    def get_gpu_time_spent_accessing_memory(self):
        return self.gpu_time_spent_accessing_memory

    def get_gpu_memory_allocated(self):
        return self.gpu_memory_allocated

    def get_gpu_power_usage(self):
        return self.gpu_power_usage
