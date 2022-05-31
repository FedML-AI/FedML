import os

import psutil
import pynvml
from wandb.sdk.internal.stats import SystemStats


class SysStats:
    def __init__(self):
        self.sys_stats_impl = SystemStats(os.getpid(), None)
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

        self.cpu_utilization = stats["cpu"]
        self.system_memory_utilization = stats["memory"]
        self.process_memory_in_use_size = stats["proc.memory.percent"]
        self.process_memory_in_use = stats["proc.memory.rssMB"]
        self.process_memory_available = stats["proc.memory.availableMB"]
        self.process_cpu_threads_in_use = stats["proc.cpu.threads"]
        self.disk_utilization = stats["disk"]
        self.network_traffic = stats["network"]["sent"] + stats["network"]["recv"]

        for stat_key, stat_value in stats.items():
            if str(stat_key).find("gpu.0.gpu") != -1:
                if self.sys_stats_impl.gpu_count == 0:
                    self.sys_stats_impl.gpu_count = 1

        gpu_mem_used = 0.0
        gpu_usage_total = 0.0
        gpu_mem_allocated = 0.0
        gpu_temperature_total = 0.0
        gpu_power_usage_total = 0.0
        for i in range(self.sys_stats_impl.gpu_count):
            gpu_mem_used += stats["gpu.{}.{}".format(i, "memory")]
            gpu_usage_total += stats["gpu.{}.{}".format(i, "gpu")]
            gpu_mem_allocated += stats["gpu.{}.{}".format(i, "memoryAllocated")]
            gpu_temperature_total += stats["gpu.{}.{}".format(i, "temp")]
            gpu_power_usage_total += stats["gpu.{}.{}".format(i, "powerPercent")]
        if self.sys_stats_impl.gpu_count >= 1:
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

    def produce_info_internal(self):
        self.cpu_utilization = psutil.cpu_percent()

        sys_mem_obj = psutil.virtual_memory()
        self.system_memory_utilization = sys_mem_obj.mem_percent
        total_mem_size = sys_mem_obj.mem_total

        process_mem_percent_total = 0.0
        process_threads_total = 0
        pids = psutil.pids()
        for pid in pids:
            process = psutil.Process(pid)
            process_mem_percent_total += process.memory_percent()
            process_threads_total += process.num_threads()

        self.process_memory_in_use_size = process_mem_percent_total
        self.process_memory_in_use = total_mem_size * process_mem_percent_total
        self.process_memory_available = total_mem_size * (
            1.0 - process_mem_percent_total
        )
        self.process_cpu_threads_in_use = process_threads_total

        disk_partitions = psutil.disk_partitions()
        disk_total_size = 0.0
        disk_used_size = 0.0
        for disk in disk_partitions:
            disk_usage = psutil.disk_usage(disk.mountpoint)
            disk_total_size += disk_usage.total
            disk_used_size += disk_usage.used
        self.disk_utilization = disk_used_size / disk_total_size

        net_io_counters = psutil.net_io_counters()
        self.network_traffic = (
            net_io_counters.bytes_sent
            - self.network_init["bytes_send"]
            + net_io_counters.bytes_recv
            - self.network_init["bytes_recv"]
        )

        gpu_device_count = pynvml.nvmlDeviceGetCount()
        gpu_mem_used = 0.0
        gpu_mem_total = 0.0
        gpu_temperature_total = 0.0
        gpu_power_usage_total = 0.0
        for i in range(gpu_device_count):
            gpu_handle = pynvml.nvmlDeviceGetHandleByIndex(i)
            gpu_mem_info = pynvml.nvmlDeviceGetMemoryInfo(gpu_handle)
            gpu_temperature_total += pynvml.nvmlDeviceGetTemperature(
                gpu_handle, pynvml.NVML_TEMPERATURE_GPU
            )
            utilz = pynvml.nvmlDeviceGetUtilizationRates(gpu_handle)
            gpu_mem_used += gpu_mem_info.used
            gpu_mem_total += gpu_mem_info.total
            gpu_power_usage_total += pynvml.nvmlDeviceGetPowerUsage(gpu_handle)
        self.gpu_utilization = utilz.gpu
        self.gpu_memory_allocated = gpu_mem_used / gpu_mem_total
        self.gpu_temp = gpu_temperature_total / gpu_device_count
        self.gpu_power_usage = gpu_power_usage_total / gpu_device_count
        self.gpu_time_spent_accessing_memory = utilz.memory
