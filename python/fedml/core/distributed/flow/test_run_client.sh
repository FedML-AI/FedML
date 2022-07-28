
RANK=$1
python test_fedml_flow.py --cf fedml_config.yaml --rank $RANK --role client --run_id ch_flow
