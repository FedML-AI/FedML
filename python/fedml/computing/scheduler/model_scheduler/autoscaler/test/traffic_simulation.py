import datetime
import random
import sys

import numpy as np

from datetime import timedelta

sys.path.insert(0, '..')  # Need to extend the path because the test script is a standalone script.
random.seed(0)


class TrafficSimulation(object):
    CONFIG_DATETIME_FORMAT = "%Y-%m-%dT%H:%M:%S"
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
    QPS_MIN_VAL = 1
    QPS_MAX_VAL = 100
    LATENCY_MIN_VAL = 1
    LATENCY_MAX_VAL = 5

    def __init__(self, start_date: datetime.datetime = None):
        if not start_date:
            self.START_DATE = datetime.datetime(2001, 1, 1, 1, 1, 1)  # 2001-01-01T01:01:01z"
        else:
            self.START_DATE = start_date

    @staticmethod
    def date_increment_sec(start_date, secs):
        timestamp = start_date + datetime.timedelta(seconds=secs)
        return timestamp

    @staticmethod
    def date_increment_min(start_date, minutes):
        timestamp = start_date + datetime.timedelta(minutes=minutes)
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
    def low_high(cls, latency=False, qps=False, num_values=1, reverse=False):
        low, high, step = None, None, None
        if latency:
            low, high = \
                (cls.LATENCY_MAX_VAL, cls.LATENCY_MIN_VAL) if reverse \
                    else (cls.LATENCY_MIN_VAL, cls.LATENCY_MAX_VAL)
            step = (cls.LATENCY_MAX_VAL - cls.LATENCY_MIN_VAL) / num_values
            step = -step if reverse else step
        if qps:
            low, high = \
                (cls.QPS_MAX_VAL, cls.QPS_MIN_VAL) if reverse \
                    else (cls.QPS_MIN_VAL, cls.QPS_MAX_VAL)
            step = (cls.QPS_MAX_VAL - cls.QPS_MIN_VAL) / num_values
            step = -step if reverse else step
        return low, high, step

    def generate_traffic(self,
                         qps_distribution,
                         latency_distribution,
                         num_values,
                         submit_request_every_x_secs=30,
                         reverse=False,
                         with_warmup=False):

        qps_values, latency_values = [], []
        q_low, q_high, q_step = self.low_high(qps=True, num_values=num_values, reverse=reverse)
        l_low, l_high, l_step = self.low_high(latency=True, num_values=num_values, reverse=reverse)
        if with_warmup:
            qps_values, latency_values = self.generate_warmup_traffic()
        if qps_distribution == "random":
            qps_values_tmp = np.round(np.random.uniform(q_low, q_high, num_values), 3).tolist()
            qps_values.extend(qps_values_tmp)
        if latency_distribution == "random":
            latency_values_tmp = np.round(np.random.uniform(l_low, l_high, num_values), 3).tolist()
            latency_values.extend(latency_values_tmp)
        if qps_distribution == "linear":
            qps_values_tmp = np.round(np.arange(q_low, q_high, q_step), 3).tolist()
            qps_values.extend(qps_values_tmp)
        if latency_distribution == "linear":
            latency_values_tmp = np.round(np.arange(l_low, l_high, l_step), 3).tolist()
            latency_values.extend(latency_values_tmp)
        if qps_distribution == "exponential":
            qps_values_tmp = np.round(np.logspace(np.log(q_low), np.log(q_high), num_values, base=np.exp(1)),
                                      3).tolist()
            qps_values.extend(qps_values_tmp)
        if latency_distribution == "exponential":
            low, high, _ = self.low_high(latency=True, num_values=num_values, reverse=reverse)
            latency_values_tmp = np.round(np.logspace(np.log(l_low), np.log(l_high), num_values, base=np.exp(1)),
                                          3).tolist()
            latency_values.extend(latency_values_tmp)

        traffic = []
        current_timestamp = self.START_DATE
        for q, l in zip(qps_values, latency_values):
            timestamp = current_timestamp.strftime(self.CONFIG_DATETIME_FORMAT)
            current_timestamp = self.date_increment_sec(
                current_timestamp, secs=submit_request_every_x_secs)
            traffic.append((timestamp, q, l))

        return traffic

    def generate_traffic_with_seasonality(self,
                                          num_values,
                                          submit_request_every_x_secs=30,
                                          with_warmup=False):

        # We import mockseries here, because mockseries is not
        # a standard package required by fedml to be installed.
        from mockseries.noise import RedNoise
        from mockseries.utils import datetime_range
        from mockseries.seasonality import SinusoidalSeasonality
        from mockseries.trend import LinearTrend

        traffic = []
        start_date = self.START_DATE
        if with_warmup:
            qps_values, latency_values = self.generate_warmup_traffic()
            current_timestamp = start_date
            for q, l in zip(qps_values, latency_values):
                timestamp = current_timestamp.strftime(self.CONFIG_DATETIME_FORMAT)
                current_timestamp = self.date_increment_sec(
                    current_timestamp, secs=submit_request_every_x_secs)
                traffic.append((timestamp, q, l))
            start_date = current_timestamp

        trend = LinearTrend(coefficient=2, time_unit=timedelta(hours=1), flat_base=100)
        seasonality = SinusoidalSeasonality(amplitude=20, period=timedelta(hours=1)) \
                      + SinusoidalSeasonality(amplitude=4, period=timedelta(hours=1))
        noise = RedNoise(mean=0, std=3, correlation=0.5)

        timeseries = trend + seasonality + noise
        ts_index = datetime_range(
            granularity=timedelta(seconds=submit_request_every_x_secs),
            start_time=start_date,
            num_points=num_values
        )
        ts_values = timeseries.generate(ts_index)

        r_min, r_max = ts_values.min(), ts_values.max()
        scale_to_qps = \
            lambda x: ((x - r_min) / (r_max - r_min)) * (self.QPS_MAX_VAL - self.QPS_MIN_VAL) + self.QPS_MIN_VAL
        scale_to_latency = \
            lambda x: ((x - r_min) / (r_max - r_min)) * (
                        self.LATENCY_MAX_VAL - self.LATENCY_MIN_VAL) + self.LATENCY_MIN_VAL

        for d, v in zip(ts_index, ts_values):
            qps = scale_to_qps(v)
            lat = scale_to_latency(v)
            ts = d.strftime(self.CONFIG_DATETIME_FORMAT)
            traffic.append((ts, qps, lat))

        return traffic
