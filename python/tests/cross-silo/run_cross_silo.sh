#!/bin/bash
set -e
WORKSPACE=$(pwd)
# PROJECT_HOME=$WORKSPACE/../../
# cd $PROJECT_HOME

cd examples/federate/cross_silo/mqtt_s3_fedavg_mnist_lr_example/custom_data_and_model

# run client(s)
RUN_ID="$(python -c "import uuid; print(uuid.uuid4().hex)")"

for client_rank in {1..2}; do
  nohup bash run_client.sh "$client_rank" "$RUN_ID" 1 > $WORKSPACE/client_"$client_rank"_out.log 2>&1 &
done

# run server
nohup bash run_server.sh "$RUN_ID" 1 > $WORKSPACE/server_out.log 2>&1 &

# after execution, we can query the progress id (pid) with: ps -ef | grep "fedml_config"
# kill the processes with: kill -9 <pid>

wait

if [ "$?" = "0" ]; then
  echo "successfully run cross-silo example"
else
  echo "failed on cross-silo example"
  exit 1
fi