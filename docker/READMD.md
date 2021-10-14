

# Build Docker Image
```
# 1. install docker engine according to https://docs.docker.com/engine/install/ubuntu/

# 2. build
sudo chmod 777 /var/run/docker.sock
docker build . -f Dockerfile.fedml


# 3. push docker to the cloud (change image-id obtained at step 2)
docker tag be3c6a946c76 fedml/fedml:1.0

docker login --username fedml
docker push fedml/fedml:1.0

# start to run docker (refer to experiments/distributed/docker/run_docker_on_single_node.sh)
```