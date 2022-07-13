# Install FedML and Prepare the Distributed Environment
```
pip install fedml
```

# Run the example (one line API)
```
# 4 means 4 processes
sh run_one_line_example.sh 4
```

# Run the example (step by step APIs)
```
sh run_step_by_step_example.sh 4
```

# Run the example (custom dataset and model)
```
sh run_custom_data_and_model_example.sh 4
```

# 
```
sh run.sh 4 "scigpu13:0,1,1,1;scigpu14:0,2,1,1"
```
mpirun -np 5 \
-host "scigpu13:3,scigpu12:2" \
/home/comp/20481896/anaconda3/envs/py36/bin/python torch_fedavg_mnist_lr_custum_data_and_model_example.py --cf config/zht_config.yaml \








