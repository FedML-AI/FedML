```sh
export ACCOUNT_ID=YOUR_USER_ID
export API_KEY=YOUR_API_KEY
sh build_and_push_pacakge.sh

#Master
fedml model device login $usr_id -p -m
#Slave
fedml model device login $usr_id -p
```