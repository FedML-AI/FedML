# 1. install docker engine according to https://docs.docker.com/engine/install/ubuntu/

# 2. build
sudo chmod 777 /var/run/docker.sock
docker build . -f Dockerfile

# 3. push docker to the cloud (change image-id obtained at step 2)
docker tag image-id fedml/fedml:0.7.13

docker login --username fedml
docker push fedml/fairfl:0.7.13