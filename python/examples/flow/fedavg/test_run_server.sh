RUN_ID=$1
python fedavg_flow.py --cf fedml_config.yaml --rank 0 --role server --run_id $RUN_ID