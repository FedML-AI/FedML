# FedML Installation on Raspberry Pi

## Run FedML with Docker (Recommended)
- Pull FedML RPI docker image
```
docker pull fedml/fedml:nvidia-jetson-l4t-ml-r32.6.1-py3
```

- Run Docker with "fedml login"
```
docker run -t -i fedml/fedml:nvidia-jetson-l4t-ml-r32.6.1-py3 /bin/bash

root@8bc0de2ce0e0:/usr/src/app# fedml login 299

```

