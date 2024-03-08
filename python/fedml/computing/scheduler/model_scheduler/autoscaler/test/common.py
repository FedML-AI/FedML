# import collections
# import conf
# import csv
# import datetime
# import json
# import random
# import requests
#
# import numpy as np
#
# DATA_PAYLOAD_SCHEMA = {
#     "timestamp": None,
#     "latency": None,
#     "qps": None,
# }
# PAYLOAD_SCHEMA = {
#     "endpoint_id": None,
#     "total_requests": None,
#     "data": [DATA_PAYLOAD_SCHEMA]
# }
# CONFIG_DATETIME_FORMAT = conf.CONFIG_DATETIME_FORMAT
# START_DATE = datetime.datetime(2001, 1, 1, 1, 1, 1) # 2001-01-01T01:01:01z"
# ENDPOINTS_STATS = collections.namedtuple(
#     'endpoints_stats', ['min_qps', 'max_qps', 'all_qps',
#                         'min_latency', 'max_latency', 'all_latency'])
#
#
# def post_request_reactive(payload):
#     r = requests.post(
#         "http://{}:{}/fedml/api/autoscaler".format(
#             conf.AUTOSCALER_HOST, conf.AUTOSCALER_PORT),
#         data=json.dumps(payload))
#     response = r.text
#     return response
#
#
# def post_request_shutdown():
#     r = requests.post(
#     "http://{}:{}/fedml/api/autoscaler/shutdown".format(
#         conf.AUTOSCALER_HOST, conf.AUTOSCALER_PORT))
#     response = r.text
#     return response
#
#
# def date_random(start_date):
#     timestamp = start_date + \
#         datetime.timedelta(days=random.randrange(365)) + \
#         datetime.timedelta(weeks=random.randrange(52)) + \
#         datetime.timedelta(hours=random.randrange(24)) + \
#         datetime.timedelta(minutes=random.randrange(60)) + \
#         datetime.timedelta(seconds=random.randrange(60))
#     return timestamp
#
# def date_increment_sec(start_date, secs):
#     timestamp = start_date + datetime.timedelta(seconds=secs)
#     return timestamp
#
# def date_increment_min(start_date, mins):
#     timestamp = start_date + datetime.timedelta(minutes=mins)
#     return timestamp
#
# def endpoints_stats(fname):
#     qps_data_range = collections.defaultdict(list)
#     latency_data_range = collections.defaultdict(list)
#     with open(fname, newline='') as csvfile:
#         reader = csv.DictReader(csvfile)
#         for row in reader:
#             endpoint_id = int(row["endpoint_id"])
#             latency = float(row["latency"])
#             qps = float(row["qps"])
#             endpoint_key = "{}".format(endpoint_id)
#             qps_data_range[endpoint_key].append(qps)
#             latency_data_range[endpoint_key].append(latency)
#
#     mean_q, min_q, max_q, all_q = [], np.inf, -np.inf, []
#     for _, q in qps_data_range.items():
#         q = [q_ for q_ in q if q_ > 0]
#         if len(q) > 0:
#             mean_q.append(np.mean(q))
#             min_q = np.min([min_q, np.min(q)])
#             max_q = np.max([max_q, np.max(q)])
#             all_q.extend(q)
#
#     mean_l, min_l, max_l, all_l = [], np.inf, -np.inf, []
#     for _, l in latency_data_range.items():
#         l = [l_ for l_ in l if l_ > 0]
#         if len(l) > 0:
#             mean_l.append(np.mean(l))
#             min_l = np.min([min_l, np.min(l)])
#             max_l = np.max([max_l, np.max(l)])
#             all_l.extend(l)
#
#     return ENDPOINTS_STATS(
#         min_qps=min_q, max_qps=max_q, all_qps=all_q,
#         min_latency=min_l, max_latency=max_l, all_latency=all_l)
