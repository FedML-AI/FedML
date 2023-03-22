# FedAvg Sequential training

In this implementation, you can conduct FL with infinite number of clients sampled per round. It can work well even when you only have several GPUs. You only need tp specify the federated_optimizer as "FedAvg_seq", and other parameters stay as the same as other examples.



# Install FedML and Prepare the Distributed Environment
```
pip install fedml
```

# Run the example (4 workers and 1 server on localhost)
```
sh run.sh 4 "localhost:5"
```

# Run the example (2 workers and 1 server on gpu1, 2 workers on gpu2)
```
sh run.sh 4 "gpu1:3;gpu2:2"
```


## GPU usage

gpu_mapping.yaml is used to define the gpu device usage. You can also use ``gpu_util_parse`` to define the gpu device usage.

Define the gpu_mapping in the gpu_mapping.yaml as:
```
gpu_mapping:
    host1: [1, 1, 1, 0]
    host2: [0, 2, 0, 0]
```
has the same effect of define ``gpu_util_parse="gpu1:1,1,1,0;gpu2:0,1,1,0"``. They both mean that on host1, gpu0, gpu1, gpu2 will be assigned to process 0, 1, 2 respectively, and gpu1 on host 2 will be assigned to process 3 and 4.





