
# run 10 workers, not using sequential
mpirun -np 11 \
-host "localhost:11" \
/home/chaoyanghe/anaconda3/envs/fedml/bin/python main.py --cf config/10workers.yaml \
--override_cmd_args




