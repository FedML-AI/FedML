import os
from os.path import expanduser
import time


class MLOpsUtils:
    _ntp_offset = None

    @staticmethod
    def calc_ntp_from_config(mlops_config):
        if mlops_config is None:
            return

        ntp_response = mlops_config.get("NTP_RESPONSE", None)
        if ntp_response is None:
            return

        # setup ntp time from the configs
        device_recv_time = int(time.time() * 1000)
        device_send_time = ntp_response.get("deviceSendTime", None)
        server_recv_time = ntp_response.get("serverRecvTime", None)
        server_send_time = ntp_response.get("serverSendTime", None)
        if device_send_time is None or server_recv_time is None or server_send_time is None:
            return

        # calculate the time offset(int)
        ntp_time = (server_recv_time + server_send_time + device_recv_time - device_send_time) // 2
        ntp_offset = ntp_time - device_recv_time

        # set the time offset
        MLOpsUtils.set_ntp_offset(ntp_offset)

    @staticmethod
    def set_ntp_offset(ntp_offset):
        MLOpsUtils._ntp_offset = ntp_offset

    @staticmethod
    def get_ntp_time():
        if MLOpsUtils._ntp_offset is not None:
            return int(time.time() * 1000) + MLOpsUtils._ntp_offset
        return int(time.time() * 1000)

    @staticmethod
    def get_ntp_offset():
        return MLOpsUtils._ntp_offset

    @staticmethod
    def write_log_trace(log_trace):
        log_trace_dir = os.path.join(expanduser("~"), "fedml_log")
        if not os.path.exists(log_trace_dir):
            os.makedirs(log_trace_dir, exist_ok=True)

        log_file_obj = open(os.path.join(log_trace_dir, "logs.txt"), "a")
        log_file_obj.write("{}\n".format(log_trace))
        log_file_obj.close()
