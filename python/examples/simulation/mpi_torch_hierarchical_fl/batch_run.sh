#!/usr/bin/env bash

GROUP_NUM=5
GROUP_METHOD="hetero"
COMM_ROUND=62 #250
GROUP_COMM_ROUND=4 # 1
TOPO_NAME="star"
CONFIG_PATH=config/mnist_lr/fedml_config_topo.yaml

group_alpha_list=(0.01 0.1 1.0)

WORKER_NUM=$(($GROUP_NUM+1))
hostname > mpi_host_file
mkdir -p batch_log
# we need to install yq (https://github.com/mikefarah/yq)
# wget https://github.com/mikefarah/yq/releases/latest/download/yq_linux_amd64 -O /usr/bin/yq && chmod +x /usr/bin/yq

yq -i ".device_args.worker_num = ${WORKER_NUM}" $CONFIG_PATH
yq -i ".device_args.gpu_mapping_key = \"mapping_config1_${WORKER_NUM}\"" $CONFIG_PATH
yq -i ".train_args.group_num = ${GROUP_NUM}" $CONFIG_PATH
yq -i ".train_args.comm_round = ${COMM_ROUND}" $CONFIG_PATH
yq -i ".train_args.group_comm_round = ${GROUP_COMM_ROUND}" $CONFIG_PATH
yq -i ".train_args.group_method = \"${GROUP_METHOD}\"" $CONFIG_PATH
yq -i ".train_args.topo_name = \"${TOPO_NAME}\"" $CONFIG_PATH

if [ "${GROUP_METHOD}" = "random" ]; then
  yq -i ".train_args.group_alpha = 0" $CONFIG_PATH
fi

if [ "${TOPO_NAME}" != "random" ]; then
  yq -i ".train_args.topo_edge_probability = 1.0" $CONFIG_PATH
fi


for group_alpha in ${group_alpha_list[@]};
do
  echo "group_alpha=$group_alpha"
  yq -i ".train_args.group_alpha = ${group_alpha}" $CONFIG_PATH

  nohup mpirun -np $WORKER_NUM \
  -hostfile mpi_host_file \
  python torch_step_by_step_example.py --cf $CONFIG_PATH \
  > batch_log/"group_alpha=$group_alpha.log"  2>&1 & echo $! >> batch_log/group_alpha.pid
  sleep 30
done

echo "Finished!"