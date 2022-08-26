

mpirun -np 9 \
-host "localhost:9" \
/home/chaoyanghe/anaconda3/envs/fedml/bin/python main.py --cf config/schedule_femnist_2.yaml \
--override_cmd_args



mpirun -np 9 \
-host "localhost:9" \
/home/chaoyanghe/anaconda3/envs/fedml/bin/python main.py --cf config/schedule_femnist.yaml \
--override_cmd_args



