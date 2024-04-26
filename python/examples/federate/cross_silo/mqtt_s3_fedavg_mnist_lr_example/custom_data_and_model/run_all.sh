#!/usr/bin/env bash

set -e
cd "$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"/

# The name of the current run.
RUN_ID=$1
if [ -z "${RUN_ID}" ]; then
  echo "Need to provide the id of the run as a string."
  exit 1
fi

# The number of workers.
WORKER_NUM=$2
if [ -z "${WORKER_NUM}" ]; then
  echo "Need to provide the number of workers you want to run the experiment for."
  exit 1
fi

# Spawn server process.
echo "Starting server"
python3 torch_server.py --cf config/fedml_config.yaml --rank 0 --role server --run_id $RUN_ID &
sleep 3  # Sleep for 3s to give the server enough time to start

# Spawn client(s) process.
# Change the number next to seq for spawning more than 1 clients.
for i in `seq $WORKER_NUM`; do
    echo "Starting client $i"
    python3 torch_client.py --cf config/fedml_config.yaml --rank $i --role client --run_id $RUN_ID &
done

# Enable CTRL+C to stop all background processes
trap "trap - SIGTERM && kill -- -$$" SIGINT SIGTERM

# Wait for all background processes to complete
wait
