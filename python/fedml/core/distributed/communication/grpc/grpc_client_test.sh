
python grpc_comm_manager.py 0 1 > ./grpc_client_log_0_1.txt 2>&1 &
python grpc_comm_manager.py 1 2 > ./grpc_client_log_1_2.txt 2>&1 &
python grpc_comm_manager.py 2 0 > ./grpc_client_log_2_0.txt 2>&1