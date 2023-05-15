#!/bin/bash
WORKSPACE=$(pwd)
PROJECT_HOME=$WORKSPACE/../../
cd $PROJECT_HOME

cd examples/federated_analytics/avg_fake_data_example

# run client(s)
RUN_ID="$(python -c "import uuid; print(uuid.uuid4().hex)")"

FA_TASK="AVG"

for client_rank in {1..2}; do
  nohup bash run_client.sh "$client_rank" "$RUN_ID" 1 > $WORKSPACE/client_"$client_rank"_out_"$FA_TASK".log 2>&1 &
done

# run server
nohup bash run_server.sh "$RUN_ID" 1 > $WORKSPACE/server_out_"$FA_TASK".log 2>&1 &


cd ../frequency_estimation_fake_data_example
FA_TASK="frequency_estimation"
RUN_ID="$(python -c "import uuid; print(uuid.uuid4().hex)")"

for client_rank in {1..2}; do
  nohup bash run_client.sh "$client_rank" "$RUN_ID" 1 > $WORKSPACE/client_"$client_rank"_out_"$FA_TASK".log 2>&1 &
done

# run server
nohup bash run_server.sh "$RUN_ID" 1 > $WORKSPACE/server_out_"$FA_TASK".log 2>&1 &


cd ../heavy_hitter_twitter_data_example
FA_TASK="heavy_hitter"
RUN_ID="$(python -c "import uuid; print(uuid.uuid4().hex)")"

for client_rank in {1..2}; do
  nohup bash run_client.sh "$client_rank" "$RUN_ID" 1 > $WORKSPACE/client_"$client_rank"_out_"$FA_TASK".log 2>&1 &
done

# run server
nohup bash run_server.sh "$RUN_ID" 1 > $WORKSPACE/server_out_"$FA_TASK".log 2>&1 &


cd ../intersection_and_cardinality_fake_data_example
FA_TASK="intersection"
RUN_ID="$(python -c "import uuid; print(uuid.uuid4().hex)")"

for client_rank in {1..2}; do
  nohup bash run_client.sh "$client_rank" "$RUN_ID" 1 > $WORKSPACE/client_"$client_rank"_out_"$FA_TASK".log 2>&1 &
done

# run server
nohup bash run_server.sh "$RUN_ID" 1 > $WORKSPACE/server_out_"$FA_TASK".log 2>&1 &


cd ../k_percentile_fake_data_example
FA_TASK="k_percentile"
RUN_ID="$(python -c "import uuid; print(uuid.uuid4().hex)")"

for client_rank in {1..2}; do
  nohup bash run_client.sh "$client_rank" "$RUN_ID" 1 > $WORKSPACE/client_"$client_rank"_out_"$FA_TASK".log 2>&1 &
done

# run server
nohup bash run_server.sh "$RUN_ID" 1 > $WORKSPACE/server_out_"$FA_TASK".log 2>&1 &


cd ../union_fake_data_example
FA_TASK="union"
RUN_ID="$(python -c "import uuid; print(uuid.uuid4().hex)")"

for client_rank in {1..2}; do
  nohup bash run_client.sh "$client_rank" "$RUN_ID" 1 > $WORKSPACE/client_"$client_rank"_out_"$FA_TASK".log 2>&1 &
done

# run server
nohup bash run_server.sh "$RUN_ID" 1 > $WORKSPACE/server_out_"$FA_TASK".log 2>&1 &

pid=$!
kill -9 $pid

# after execution, we can query the progress id (pid) with: ps -ef | grep "fedml_config"
# kill the processes with: kill -9 <pid>