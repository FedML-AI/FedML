
import logging
import ntplib
import time
import datetime


class MLOpsUtils:
    @staticmethod
    def get_ntp_port():
        """
        Get the ntp server port
        """
        # default port is the name of the service but lookup would fail
        # if the /etc/services file is missing. In that case, fallback to numeric
        import socket
        host = 'time.aws.com'
        port = 'ntp'
        DEFAULT_PORT_NUM = 123
        try:
            socket.getaddrinfo(host, port)
        except socket.gaierror:
            port = DEFAULT_PORT_NUM

        return port

    @staticmethod
    def get_ntp_offset():
        cnt = 0
        ntp_server_url = 'time.aws.com'
        while True:  # try until we get time offset
            try:
                ntp_client = ntplib.NTPClient()
                req_args = {
                    'host': ntp_server_url,
                    'port': MLOpsUtils.get_ntp_port(),
                    'version': 10,
                }
                ntp_time = datetime.datetime.utcfromtimestamp(
                    ntp_client.request(**req_args).tx_time).timestamp()
                loc_computer_time = time.time()
                offset = ntp_time - loc_computer_time
                return offset
            except Exception as e:
                cnt += 1
                time.sleep(1)
                if cnt >= 3:
                    break
        return None
