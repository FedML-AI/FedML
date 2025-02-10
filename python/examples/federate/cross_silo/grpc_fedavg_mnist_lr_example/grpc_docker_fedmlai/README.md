
# Introduction
In this working example, we will run 1 aggregation server and 2 clients on the same machine using Docker + gRPC and we will use the TensorOpera.ai platform to run the FL job. 

# gRPC Configuration File
The content of the gRPC configuration file is as follows:
```
eid,rank,grpc_server_ip,grpc_server_port,ingress_ip
0,0,0.0.0.0,8890,fedml_server
1,1,0.0.0.0,8899,fedml_client_1
2,2,0.0.0.0,8898,fedml_client_2
```
The ingress_ip variable refers to the name of the container that we assign to either the server or the client, as we discuss in detail below:


# Docker Configuration
Before creating any docker container one our machine, we need to pull the latest fedml image (e.g., `fedml:v090`) and ensure that all spawned containers can communicate to each other through a network bridge (e.g., `fedml_grpc_network`).  
Specifically, what you need to do is:
```bash
docker pull fedml:v090
docker network create fedml_grpc_network
``` 

Once these two steps are configured we can start 1 aggregation server and 2 clients (without using a GPU) and register them using our <FEDML_API_KEY> with the fedml platform as follows:

```bash
# Server
docker run -it -p 8890:8890 --entrypoint /bin/bash --name fedml_server --network fedml_grpc_network fedml:dev090
redis-server --daemonize yes
source /fedml/bin/activate
fedml login -s <FEDML_API_KEY>
```

```bash
# Client 1
docker run -it -p 8891:8891 --entrypoint /bin/bash --name fedml_client_1 --network fedml_grpc_network fedml:dev090
redis-server --daemonize yes
source /fedml/bin/activate
fedml login -c <FEDML_API_KEY>
```

```bash
# Client-2
docker run -it -p 8892:8892 --entrypoint /bin/bash --name fedml_client_2 --network fedml_grpc_network fedml:dev090
redis-server --daemonize yes
source /fedml/bin/activate
fedml login -c <FEDML_API_KEY>
```

Then we only need to compile our job and submit to our dockerb-based cluster as it is also discussed in detail in the official TensorOpera documentation: https://tensoropera.ai/octopus/userGuides

