
import logging
import ntplib
import time
import datetime


class MLOpsUtils:

    @staticmethod
    def get_ntp_offset():
        cnt = 0
        ntp_server_url = 'time.aws.com'
        while True:  # try until we get time offset
            try:
                ntp_client = ntplib.NTPClient()
                ntp_time = datetime.datetime.utcfromtimestamp(
                    ntp_client.request(ntp_server_url, 10).tx_time).timestamp()
                loc_computer_time = time.time()
                offset = ntp_time - loc_computer_time
                return offset
            except Exception as e:
                cnt += 1
                time.sleep(1)
                if cnt >= 3:
                    logging.info(f"Cannot Connect To NTP Server: {ntp_server_url}")
                    break
        return None
