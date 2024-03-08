# import sys
# sys.path.insert(0, '..') # Need to extend the path because the test script is a standalone script.
#
# import common as common
# import conf
# import datetime as dt
# import matplotlib.pyplot as plt
# import multiprocessing
#
# from autoscaler.autoscaler_app import app
# from matplotlib.dates import DateFormatter
# from stress_test import StressTest
#
# def plot_qps_vs_latency_vs_scale(stress_tests):
#     for test in stress_tests:
#         endpoint_id = test.endpoint_id
#         payloads = test.payloads
#         responses = test.responses
#
#     # plot
#     fig, ax = plt.subplots(figsize=(30, 10))
#     ts = [x["timestamp"] for datum in payloads for x in datum["data"]]
#     ts = [dt.datetime.strptime(t, common.CONFIG_DATETIME_FORMAT) for t in ts]
#     ax.xaxis.set_major_formatter(DateFormatter("%H:%M:%S"))
#     qps = [x["qps"] for datum in payloads for x in datum["data"]]
#     latency = [x["latency"] for datum in payloads for x in datum["data"]]
#     running_instances = []
#     for res in responses:
#         scale_op = 0
#         if res["allocate_instance"] == "1":
#             scale_op = 1
#         if res["free_instance"] == "1":
#             scale_op = -1
#
#         # If it is the first time we populate the list then,
#         # assign the default scale operation, else use
#         # augment according to scale_op value: -1, 0, +1
#         if not running_instances:
#             running_instances.append(scale_op)
#         else:
#             running_instances.append(running_instances[-1] + scale_op)
#
#     ax.plot_date(ts, qps, color='red', fmt="8--", linewidth=0.5, label="QPS")
#     ax.plot_date(ts, latency, color='purple', fmt="p--", linewidth=0.5, label="Latency")
#     ax.plot_date(ts, running_instances, color='green', fmt="*", linewidth=0.5, label="Instances")
#     ax.set_xlabel("Timestamp")
#     # ax.set_ylabel("QPS")
#     plt.xticks(rotation=0)
#     for label in ax.xaxis.get_ticklabels():
#         label.set_rotation(45)
#     ax.grid(True)
#     plt.legend()
#     plt.savefig('plot/test.png', bbox_inches='tight')
#
#
# if __name__ == "__main__":
#
#     multiprocessing.set_start_method('spawn', force=True)
#
#     # We set "debug": False to avoid double
#     # initialization of the Flask App.
#     server = multiprocessing.Process(target=app.run, kwargs={
#         "host": conf.AUTOSCALER_HOST,
#         "port": conf.AUTOSCALER_PORT,
#         "debug": False,
#         "use_reloader": False})
#     server.start()
#
#     thread_pool_workers = 10
#     for num_endpoints in [1]:
#         for requests_per_endpoint in [1]:
#             for qps_distribution in ["random"]:
#                 for latency_distribution in ["random"]:
#                     reactive_stress_tests = \
#                         StressTest.stress_test_reactive(
#                             num_endpoints=num_endpoints,
#                             warmup_requests_per_endpoint=100,
#                             submit_request_every_x_secs=60,
#                             requests_per_endpoint=requests_per_endpoint,
#                             thread_pool_workers=thread_pool_workers,
#                             qps_distribution=qps_distribution,
#                             latency_distribution=latency_distribution)
#                     plot_qps_vs_latency_vs_scale(reactive_stress_tests)
#
#     # Send shutdown signal
#     common.post_request_shutdown()
#     # Terminate the process.
#     server.terminate()
#     server.join()
