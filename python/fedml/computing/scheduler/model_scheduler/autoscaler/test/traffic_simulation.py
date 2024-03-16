import random
import sys

import datetime
import numpy as np

import fedml.computing.scheduler.model_scheduler.autoscaler.conf as conf

sys.path.insert(0, '..')    # Need to extend the path because the test script is a standalone script.
random.seed(0)


class TrafficSimulation(object):

    CONFIG_DATETIME_FORMAT = conf.CONFIG_DATETIME_FORMAT
    START_DATE = datetime.datetime(2001, 1, 1, 1, 1, 1)  # 2001-01-01T01:01:01z"
    WARMUP_ENDPOINT_QPS = [
        4.538, 3.615, 3.276, 5.176, 5.73, 2.998, 2.791, 4.089, 3.913, 5.199, 4.557,
        5.308, 3.521, 5.398, 5.336, 5.74, 3.453, 5.678, 3.674, 3.405, 5.099, 3.146,
        5.128, 5.813, 3.304, 5.22, 3.304, 5.299
    ]
    WARMUP_ENDPOINT_LATENCY = [
        0.122, 0.112, 0.108, 0.101, 0.096, 0.092, 0.089, 0.088, 0.086, 0.082, 0.080,
        0.080, 0.079, 0.077, 0.078, 0.079, 0.077, 0.078, 0.078, 0.085, 0.060, 0.057,
        0.086, 0.083, 0.081, 0.079, 0.095, 0.065
    ]

    @staticmethod
    def date_increment_sec(start_date, secs):
        timestamp = start_date + datetime.timedelta(seconds=secs)
        return timestamp

    @staticmethod
    def date_increment_min(start_date, mins):
        timestamp = start_date + datetime.timedelta(minutes=mins)
        return timestamp

    @classmethod
    def generate_warmup_traffic(cls, num_values=100):
        # We will select values randomly within the IQR (Inter Quartile Range)
        # range of distribution IQR = [Q3-Q1]  Q1 | Median | Q3
        # to avoid small and large outliers both for qps and latency
        qps_q1 = np.percentile(cls.WARMUP_ENDPOINT_QPS, 25)
        qps_q3 = np.percentile(cls.WARMUP_ENDPOINT_QPS, 75)
        qps_values = np.round(np.random.uniform(qps_q1, qps_q3, num_values), 3)
        qps_values = qps_values.tolist()

        latency_q1 = np.percentile(cls.WARMUP_ENDPOINT_LATENCY, 25)
        latency_q3 = np.percentile(cls.WARMUP_ENDPOINT_LATENCY, 75)
        latency_values = np.round(np.random.uniform(latency_q1, latency_q3, num_values), 3)
        latency_values = latency_values.tolist()

        return qps_values, latency_values

    @classmethod
    def generate_traffic(cls,
                         qps_distribution,
                         latency_distribution,
                         num_values,
                         submit_request_every_x_secs=30,
                         with_warmup=True):

        qps_values, latency_values = [], []
        if with_warmup:
            qps_values, latency_values = cls.generate_warmup_traffic()
        if qps_distribution == "random":
            qps_values_tmp = np.round(np.random.uniform(1, 100, num_values), 3).tolist()
            qps_values.extend(qps_values_tmp)
        if latency_distribution == "random":
            latency_values_tmp = np.round(np.random.uniform(1, 5, num_values), 3).tolist()
            latency_values.extend(latency_values_tmp)
        if qps_distribution == "linear":
            step = (100 - 1) / num_values
            qps_values_tmp = np.round(np.arange(1, 100, step), 3).tolist()
            qps_values.extend(qps_values_tmp)
        if latency_distribution == "linear":
            step = (5 - 1) / num_values
            latency_values_tmp = np.round(np.arange(1, 5, step), 3).tolist()
            latency_values.extend(latency_values_tmp)
        if qps_distribution == "exponential":
            qps_values_tmp = np.round(np.logspace(np.log(1), np.log(100), num_values, base=np.exp(1)), 3).tolist()
            qps_values.extend(qps_values_tmp)
        if latency_distribution == "exponential":
            latency_values_tmp = np.round(np.logspace(np.log(1), np.log(5), num_values, base=np.exp(1)), 3).tolist()
            latency_values.extend(latency_values_tmp)

        traffic = []
        current_timestamp = cls.START_DATE
        for q, l in zip(qps_values, latency_values):
            timestamp = current_timestamp.strftime(conf.CONFIG_DATETIME_FORMAT)
            current_timestamp = cls.date_increment_sec(
                current_timestamp, secs=submit_request_every_x_secs)
            traffic.append((timestamp, q, l))

        return traffic
