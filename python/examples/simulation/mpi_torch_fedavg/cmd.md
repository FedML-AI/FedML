mpirun -np 9 \
-host "localhost:9" \
/home/chaoyanghe/anaconda3/envs/fedml/bin/python main.py --cf config/optim_exp.yaml \


mpirun -np 9 \
-host "localhost:9" \
/home/chaoyanghe/anaconda3/envs/fedml/bin/python main.py --cf config/debug.yaml \




# run 10 workers, not using sequential
mpirun -np 11 \
-host "localhost:11" \
/home/chaoyanghe/anaconda3/envs/fedml/bin/python main.py --cf config/10workers.yaml \













