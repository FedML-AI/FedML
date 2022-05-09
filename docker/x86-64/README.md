1. install docker engine according to https://docs.docker.com/engine/install/ubuntu/

2. build
```
sudo chmod 777 /var/run/docker.sock
docker build . -f Dockerfile
# Successfully built c56e5f90d546
```

3. push docker to the cloud (change image-id obtained at step 2)

```
# c56e5f90d546 is the docker image ID obtained at step 2
docker tag c56e5f90d546 fedml/fedml:cuda-11.6.0-devel-ubuntu20.04

docker login --username fedml
docker push fedml/fedml:cuda-11.6.0-devel-ubuntu20.04
```
