# FedAvg Sequential training

In this implementation, you can conduct FL with infinite number of clients sampled per round. It can work well even when you only have several GPUs. You only need tp specify the federated_optimizer as "FedAvg_seq", and other parameters stay as the same as other examples.



# Install FedML and Prepare the Distributed Environment
```
pip install fedml
```

# Run the example (custom dataset and model)
```
sh run_custom_data_and_model_example.sh 4
```

```
sh run.sh 4 "scigpu13:0,1,1,1;scigpu14:0,2,1,1"
```
mpirun -np 5 \
-host "scigpu13:3,scigpu12:2" \
python torch_fedavg_mnist_lr_custum_data_and_model_example.py --cf config/zht_config.yaml \








