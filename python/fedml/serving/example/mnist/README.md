Device Login
```sh
#Master
fedml model device login $usr_id -p -m
#Slave
fedml model device login $usr_id -p
```
Deploy
```sh
export API_KEY=YOUR_API_KEY
export MASTER_DEVICE_ID=YOUR_MASTER_DEVICE_ID
export WORKER_DEVICE_ID=YOUR_WORKER_DEVICE_ID

sh one_line_deploy.sh
```