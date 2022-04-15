# FedML GPU mapping configuration "gpu_mapping.yaml"
We can define the topology of distributed training with `mpi_host_file` and `gpu_mapping.yaml`.

### MPI HOST FILE
The mpi_host_file format should be as the normal mpi_host_file. Here are two examples:
```
localhost:10
```
and 
```
host1:10
host2:10
```

Different MPI version needs different format, maybe your MPI need this format:
```
localhost slots=10
```
```
host1 slots=10
host2 slots=10
```

### GPU MAPPING
The gpu_mapping.yaml file define the mapping between process with GPUs, i.e. which process uses which GPU.

#### Process-GPU mapping format definition
You can define a cluster containing multiple GPUs within multiple machines by defining `gpu_mapping.yaml` as follows:
```
config_cluster0:
    host_name_node0: [num_of_processes_on_GPU0, num_of_processes_on_GPU1, num_of_processes_on_GPU2, num_of_processes_on_GPU3, ..., num_of_processes_on_GPU_n]

    host_name_node1: [num_of_processes_on_GPU0, num_of_processes_on_GPU1, num_of_processes_on_GPU2, num_of_processes_on_GPU3, ..., num_of_processes_on_GPU_n]
....
     host_name_node_m: [num_of_processes_on_GPU0, num_of_processes_on_GPU1, num_of_processes_on_GPU2, num_of_processes_on_GPU3, ..., num_of_processes_on_GPU_n]
```
The above *.yaml file defines a cluster with m machines (node) and n GPUs in each machine. Each GPU can hold multiple processes, depending on your own design. Note that running multiple processes within one GPU may hurt the training speed but it can fully utilize the remaining GPU memory. So only when your experiment is not time-sensitive but you also hope to run more clients in FL, you can try to define more than 1 process in a single GPU.

You can pass `--gpu_mapping_file gpu_mapping.yaml --gpu_mapping_key config_cluster0` into the `main.py`. Here, the `config_cluster0` is just a string name for `main.py` to find its configuration. We can define any name we like.

#### Example for different number of GPUs in different machine
If you have a different number of GPUs on one different machine, you can define your gpu_mapping.yaml like this:
```
config_11:
    host1: [2, 2]
    host2: [1, 1, 1]
    host3: [1, 1, 1, 1]
```
This example is also used for 11 process. But the mapping is different: Server process -- host1:GPU:0, client 1 -- host1:GPU:0, client 2 -- host1:GPU:1, client 3 -- host1:GPU:1, client 4 -- host2:GPU:0, client 5 -- host2:GPU:1, client 6 -- host2:GPU:2, client 7 -- host3:GPU:0, client 8 -- host3:GPU:1, client 9 -- host3:GPU:2, client 10 -- host3:GPU:3

#### Skip some GPU devices inside a machine
Sometimes one may want to use some GPUs in one machine, instead of all GPUs. Then you can use this:
```
config_11:
    host1: [0, 2]
    host2: [1, 0, 1]
    host3: [1, 1, 0, 1]
    host4: [0, 1, 0, 0, 0, 1, 0, 2]
```
Now the mapping become: Server process -- host1:GPU:1, client 1 -- host1:GPU:1, client 2 -- host1:GPU:0, client 3 -- host1:GPU:2, client 4 -- host3:GPU:0, client 5 -- host3:GPU:1, client 6 -- host3:GPU:3, client 7 -- host4:GPU:1, client 8 -- host4:GPU:6, client 9 -- host4:GPU:7, client 10 -- host4:GPU:7.

#### Maintain all your mappings in a single file
Normally, we use multiple GPU clusters to run our experiments, so it is good to manage all your mappings in a single file.
```
config_11:
    host1: [3, 2, 3, 3]
config_21:
    host1: [6, 5, 5, 5]
config2_21:
    host1: [3, 2, 3, 3]
    host2: [2, 2, 3, 3]
config3_21:
    host1: [2, 2, 2, 2]
    host2: [2, 2, 2, 2]
    host3: [2, 2, 0, 0]
```
So when running experiments in different clusters, just pass the right gpu_mapping_map into the main.py. This configuration can avoid many yaml files existing.

Note: If you do not pass `gpu_mapping.yaml` into `main.py`, all processes will use CPU.


