
# PyTorch DDP demo
```
# sh run_ddp.sh 8 2 0 192.168.11.1 11111
nohup sh run_ddp.sh 8 2 0 192.168.11.1 11111 > ./machine1.txt 2>&1 &
nohup sh run_ddp.sh 8 2 1 192.168.11.1 11111 > ./machine2.txt 2>&1 &

nohup sh run_ddp.sh 8 1 0 127.0.0.1 11111 > ./machine1.txt 2>&1 &
```

```
# kill all processes
kill $(ps aux | grep "ddp_demo.py" | grep -v grep | awk '{print $2}')
```