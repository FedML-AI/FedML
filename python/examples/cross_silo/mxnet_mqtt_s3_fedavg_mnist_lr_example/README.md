## Training Script

Your should install mxnet lib by the following command:
1) On Linux and MacOS x86 platform:
   pip install fedml[mxnet]
2) On Windows and Mac M1 platform, you should build mxnet lib from source:
   [https://mxnet.apache.org/get_started/build_from_source](https://mxnet.apache.org/get_started/build_from_source)

Run training and aggregator with MXNet framework.

At the client side, the client ID (a.k.a rank) starts from 1.
Please also modify config/fedml_config.yaml, changing the `worker_num` the as the number of clients you plan to run.

At the server side, run the following script:
```
bash run_server.sh your_run_id
```

For client 1, run the following script:
```
bash run_client.sh 1 your_run_id
```
For client 2, run the following script:
```
bash run_client.sh 2 your_run_id
```
Note: please run the server first.

## A Better User-experience with FedML MLOps (open.fedml.ai)
To reduce the difficulty and complexity of these CLI commands. We recommend you to use our MLOps (open.fedml.ai).
FedML MLOps provides:
- Install Client Agent and Login
- Inviting Collaborators and group management
- Project Management
- Experiment Tracking (visualizing training results)
- monitoring device status
- visualizing system performance (including profiling flow chart)
- distributed logging
- model serving

## Notes:
If you meet the following issue on Ubuntu Linux when run this example:
libquadmath.so.0: cannot open shared object file: No such file or directory

you should install the libquadmath0 package by the following command:
sudo apt-get install libquadmath0