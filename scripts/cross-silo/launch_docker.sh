WORKSPACE=/home/ec2-user/FedML

# synchronize source code
sh sync_code/async_source_code_to_single_node.sh trpc-client0
sh sync_code/async_source_code_to_single_node.sh trpc-client1
sh sync_code/async_source_code_to_single_node.sh trpc-client2
sh sync_code/async_source_code_to_single_node.sh trpc-client3
sh sync_code/async_source_code_to_single_node.sh trpc-client4
sh sync_code/async_source_code_to_single_node.sh trpc-client5
sh sync_code/async_source_code_to_single_node.sh trpc-client6
sh sync_code/async_source_code_to_single_node.sh trpc-client7

# synchronize dataset


# synchronize configuration
HOST_FILE_PDSH_ACCOUNT=hostfile_account_pdsh
HOST_FILE=hostfile

NODE_NUM=8
MASTER_IP=172.31.46.221

# start processes in each node
pdsh -w ^$HOST_FILE_PDSH_ACCOUNT -R ssh "sudo pkill python; \
cd $WORKSPACE/scripts/cross-silo; \
sh run_docker_on_single_node.sh $WORKSPACE $NODE_NUM $MASTER_IP $HOST_FILE"


#pdsh -w ^hostfile -R ssh "sudo pip3 install setproctitle torchvision"