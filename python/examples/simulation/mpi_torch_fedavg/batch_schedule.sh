

# mpirun -np 9 \
# -host "localhost:9" \
# /home/chaoyanghe/anaconda3/envs/fedml/bin/python main.py --cf config/schedule_femnist_2.yaml \
# --override_cmd_args



mpirun -np 9 \
-host "localhost:9" \
/home/chaoyanghe/anaconda3/envs/fedml/bin/python main.py --cf config/schedule_femnist.yaml \
--override_cmd_args




# mpirun -np 9 \
# -host "localhost:9" \
# /home/chaoyanghe/anaconda3/envs/fedml/bin/python main.py --cf config/schedule_stackoverflow.yaml \
# --override_cmd_args


# mpirun -np 9 \
# -host "localhost:9" \
# /home/chaoyanghe/anaconda3/envs/fedml/bin/python main.py --cf config/schedule_stackoverflow_2.yaml \
# --override_cmd_args


# mpirun -np 5 \
# -host "localhost:5" \
# /home/chaoyanghe/anaconda3/envs/fedml/bin/python main.py --cf config/schedule_reddit.yaml \
# --override_cmd_args


# mpirun -np 5 \
# -host "localhost:5" \
# /home/chaoyanghe/anaconda3/envs/fedml/bin/python main.py --cf config/schedule_reddit_2.yaml \
# --override_cmd_args

