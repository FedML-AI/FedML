# Fed Launch for all algorithms

This launch scripts are implemented for launching all algorithms. Some algorithms are inclueded in the main.py have not been merged into it, due to they are under debugging.

## How to use

You should define your own `mpi_host_file` and `gpu_util.yaml`. You can put them anywhere you want, as long as you pass the right file paths to the main.py

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

### GPU UTIL
The gpu_util.yaml file define the mapping between process with gpus, i.e. which process uses which gpu.

You can define a gpu_util.yaml like this:
```
config_11:
    host1: [3, 2, 3, 3]
```
Then you can pass --gpu_util_file gpu_util.yaml --gpu_util_key config_11 into the main.py. Here, the config_11 is just a name, and do not have any meaning that the main.py need to parse. I adding the postfix as '_11' is just for meanning it is used for 11 processes. 


This example is used for 11 process, in which 1 server process and 10 client processes. The server process will run on host1:GPU:0, client 1 and 2 will run on host1:GPU:0, client 3 and 4 will run on host1:GPU:1, client 5, 6 and 7 will run on host1:GPU:2, client 8, 9 and 10 will run on host1:GPU:3. If you have different number of GPUs on one different machines, you can define you gpu_util.yaml like this:
```
config_11:
    host1: [2, 2]
    host2: [1, 1, 1]
    host3: [1, 1, 1, 1]
```
This example is also used for 11 process. But the mapping is different: Server process -- host1:GPU:0, client 1 -- host1:GPU:0, client 2 -- host1:GPU:1, client 3 -- host1:GPU:1, client 4 -- host2:GPU:0, client 5 -- host2:GPU:1, client 6 -- host2:GPU:2, client 7 -- host3:GPU:0, client 8 -- host3:GPU:1, client 9 -- host3:GPU:2, client 10 -- host3:GPU:3


And you also can add many mappings in one yaml file like this:
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
Just remember to pass the right gpu_util_key into the main.py. This configuration can avoid many yaml files existing.

If you do not pass gpu_util_file into main.py, all processes will use CPU.



## Running scripts
You can find some example running scripts in the /experiment_scripts for the according algorithms.













