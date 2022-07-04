fedml build -t client -sf . -ep torch_server.py -cf config -df ./mlops
fedml build -t server -sf . -ep torch_client.py -cf config -df ./mlops
