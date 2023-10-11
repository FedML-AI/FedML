import os
from os.path import expanduser
import time


class MLOpsUtils:
    """
    Class for MLOps utilities.
    """
    _ntp_offset = None
    BYTES_TO_GB = 1 / (1024 * 1024 * 1024)

    @staticmethod
    def calc_ntp_from_config(mlops_config):
        """
        Calculate NTP time offset from MLOps configuration.

        Args:
            mlops_config (dict): MLOps configuration containing NTP response data.

        Returns:
            None: If the necessary NTP response data is missing or invalid.
        """
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
        ntp_time = (server_recv_time + server_send_time +
                    device_recv_time - device_send_time) // 2
        ntp_offset = ntp_time - device_recv_time

        # set the time offset
        MLOpsUtils.set_ntp_offset(ntp_offset)

    @staticmethod
    def set_ntp_offset(ntp_offset):
        """
        Set the NTP time offset.

        Args:
            ntp_offset (int): The NTP time offset.
        """
        MLOpsUtils._ntp_offset = ntp_offset

    @staticmethod
    def get_ntp_time():
        """
        Get the current time adjusted by the NTP offset.

        Returns:
            int: The NTP-adjusted current time in milliseconds.
        """
        if MLOpsUtils._ntp_offset is not None:
            return int(time.time() * 1000) + MLOpsUtils._ntp_offset
        return int(time.time() * 1000)

    @staticmethod
    def get_ntp_offset():
        """
        Get the current NTP time offset.

        Returns:
            int: The NTP time offset.
        """
        return MLOpsUtils._ntp_offset

    @staticmethod
    def write_log_trace(log_trace):
        """
        Write a log trace to a file in the "fedml_log" directory.

        Args:
            log_trace (str): The log trace to write.
        """
        log_trace_dir = os.path.join(expanduser("~"), "fedml_log")
        if not os.path.exists(log_trace_dir):
            os.makedirs(log_trace_dir, exist_ok=True)

        log_file_obj = open(os.path.join(log_trace_dir, "logs.txt"), "a")
        log_file_obj.write("{}\n".format(log_trace))
        log_file_obj.close()
