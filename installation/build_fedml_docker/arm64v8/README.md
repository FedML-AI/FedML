![docker-jetson-workflow.jpg](docker-jetson-workflow.jpg)

- 1.Install docker engine according to https://docs.docker.com/engine/install/ubuntu/

- 2.Setting Up ARM Emulation on x86

https://www.stereolabs.com/docs/docker/building-arm-container-on-x86/

```
sudo apt-get install qemu binfmt-support qemu-user-static
```

- 3.Install Docker on Raspberry Pi

- 4. build
```
sudo chmod 777 /var/run/docker.sock
docker build . -f Dockerfile
```

- 5.push docker to the cloud (change image-id obtained at step 2)

```
docker tag image-id fedml/fedml:0.7.27

docker login --username fedml
docker push fedml/fedml:0.7.27
```

- 6.Docker Run
