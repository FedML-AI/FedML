# import fedml
# import json
#
# from flask import Flask, request, jsonify
# from jsonschema import validate as jsonschema_validate
#
# from autoscaler import FedMLAutoscaler
#
#
# PAYLOAD_SCHEMA = {
#     "$schema": "https://json-schema.org/draft/2019-09/schema",  # This must be specified.
#     "type": "object",
#     "properties": {
#         "endpoint_id": {
#             "type": "integer"
#         },
#         "total_requests": {
#             "type": "integer"
#         },
#         "policy": {
#             "type": "string"
#         },
#         "data": {
#             "type": "array",
#             "items": {
#                 "type": "object",
#                 "properties": {
#                     "timestamp": {
#                        "type": "string",
#                     },
#                     "latency": {
#                        "type": "number",
#                     },
#                     "qps": {
#                        "type": "number",
#                     },
#                 },
#                 "required": ["timestamp", "latency", "qps"],
#                 "additionalProperties": False  # Enforcing that no additional attributes will be provided.
#             }
#         }
#     },
#     "required": ["endpoint_id", "total_requests", "policy"],
#     "additionalProperties": False,  # Enforcing that no additional attributes will be provided.
# }
#
#
# # HTTP server
# app = Flask(__name__)
# FedMLAutoscaler.get_instance().set_config_version()
#
# @app.route('/fedml/api/autoscaler', methods=['POST'])
# def autoscaler_prediction():
#     """
#         endpoint_id: the unique identifier of the endpoint
#         total_requests: the number of submitted requests from the beginning of time (from when the endpoint was created)
#         policy: whether to trigger the ("reactive" or "predictive") policy
#         data: one or a list/sequence of timestamped records
#
#         Below is an example of an incoming payload schema:
#             {
#                 "endpoint_id": int,
#                 "total_requests": int,
#                 "policy": str
#                 "data": [{
#                             "timestamp": YYYY-MM-DDTHH:MM:SSZ,
#                             "latency": float,
#                             "qps": float
#                 }, ... {}]
#             }
#     """
#
#     try:
#         # Set default values for the return types.
#         error_code, error_message, allocate_instance, free_instance, trigger_predictive = [None] * 5
#         _data = json.loads(request.get_data(as_text=True))
#
#         # Make sure the inocming schema conforms to the expected schema.
#         jsonschema_validate(instance=_data, schema=PAYLOAD_SCHEMA)
#
#         # Extract values from incoming payload.
#         request_endpoint_id = _data.get("endpoint_id", None)
#         request_total_requests = _data.get("total_requests", None)
#         request_policy = _data.get("policy", None)
#         request_data = _data.get("data", None)
#
#         if request_policy.lower() == "reactive":
#             allocate_instance, free_instance, trigger_predictive, error_code, error_message = \
#                 FedMLAutoscaler.get_instance().reactive(request_endpoint_id, request_total_requests, request_data)
#         elif request_policy.lower() == "predictive":
#             allocate_instance, free_instance, trigger_predictive, error_code, error_message = \
#                 FedMLAutoscaler.get_instance().predictive(request_endpoint_id, request_total_requests, request_data)
#
#     except Exception as e:
#         logger.exception(e)
#         error_code = fedml.api.constants.ApiConstants.ERROR_CODE[fedml.api.constants.ApiConstants.ERROR]
#         error_message = str(e)
#     finally:
#         return jsonify({"error_code": error_code,
#                         "error_message": error_message,
#                         "allocate_instance": allocate_instance,
#                         "free_instance": free_instance,
#                         "trigger_predictive": trigger_predictive})
#
# def shutdown_server():
#     func = request.environ.get('werkzeug.server.shutdown')
#     if func is None:
#         raise RuntimeError('Not running with the Werkzeug Server')
#     func()
#
# @app.route('/fedml/api/autoscaler/shutdown', methods=['POST'])
# def shutdown():
#     try:
#         shutdown_server()
#         return jsonify({"shutdown": 1, "message": ""})
#     except Exception as e:
#         logger.exception(e)
#         return jsonify({"shutdown": 0, "message": str(e)})
#