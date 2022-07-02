fedml build -t client -sf . -ep torch_fedml_object_detection_client.py -cf config/fedml_object_detection.yaml -df ../mlops
fedml build -t server -sf . -ep torch_fedml_object_detection_server.py -cf config/fedml_object_detection.yaml -df ../mlops
