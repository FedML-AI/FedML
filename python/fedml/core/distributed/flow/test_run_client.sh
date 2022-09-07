
RANK=$1
RUN_ID=$2
python test_fedml_flow.py --cf fedml_config.yaml --rank $RANK --role client --run_id $RUN_ID
