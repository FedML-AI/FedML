## functionality of each file

**visualization_stats** prints basic stats for input partition method 

**visualization_heatmap_unsort** is the draw a heatmap. Each cell of the heatmap is the jensen-shannon distance of the corresponding client. For example (i,j) is the the distance between client_i and client_j
The larger the difference the larger the distance 

**visualization_heatmap_label** is a heatmap of the distribution of all labels within each clients

**visualization_distplot** draws the distplots corresponding to all the label skew partition methods and
put all the graphs in the same plot

**visualization_quantity_distplot** draws the distplots corresponding to all the quantity skew partition methods and put all the graphs in the same plot

## Usage
``` bash
DATA_DIR=~/fednlp_data/

python -m data.advanced_partition.util.visualization_heterogeneity \
--partition_name  'niid_label_clients=100_alpha=1.0' \
--partition_file ${DATA_DIR}/partition_files/20news_partition.h5 \
--data_file ${DATA_DIR}/data_files/20news_data.h5 \
--task_name 20news \
--figure_path ${DATA_DIR}/advanced_partition/heatmap_figure \
--task_type text_classification 

python -m data.advanced_partition.util.visualization_heterogeneity \
--partition_name  'niid_cluster_clients=100_alpha=1.0' \
--partition_file ${DATA_DIR}/partition_files/wikiner_partition.h5 \
--data_file ${DATA_DIR}/data_files/wikiner_data.h5 \
--task_name wikiner \
--figure_path ${DATA_DIR}/advanced_partition/heatmap_figure \
--task_type name_entity_recognition

python -m data.advanced_partition.util.visualization_stats \
--partition_name kmeans_10 \
--partition_file ${DATA_DIR}/partition_files/wikiner_partition.h5 \
--task_name wikiner \
--figure_path ${DATA_DIR}/advanced_partition/heatmap_figure

python -m data.advanced_partition.util.visualization_heatmap_unsort \
--partition_name 'niid_cluster_clients=100_alpha=1.0' \
--client_num 100 \ 
--partition_file ${DATA_DIR}/partition_files/wikiner_partition.h5 \
--data_file ${DATA_DIR}/data_files/wikiner_data.h5 \
--task_name wikiner \
--figure_path ${DATA_DIR}/advanced_partition/heatmap_figure \
--task_type name_entity_recognition

python -m data.advanced_partition.util.visualization_heatmap_unsort \
--partition_name  'niid_label_clients=100_alpha=1.0' \
--client_num 100 \
--partition_file ${DATA_DIR}/partition_files/agnews_partition.h5 \
--data_file ${DATA_DIR}/data_files/agnews_data.h5 \
--task_name agnews\
--figure_path ${DATA_DIR}/advanced_partition/heatmap_figure \
--task_type text_classification 

python -m data.advanced_partition.util.visualization_heatmap_label \
--partition_name 'niid_cluster_clients=100_alpha=1.0' \
--client_num 100 \ 
--partition_file ${DATA_DIR}/partition_files/wikiner_partition.h5 \
--data_file ${DATA_DIR}/data_files/wikiner_data.h5 \
--task_name wikiner \
--figure_path ${DATA_DIR}/advanced_partition/heatmap_figure \
--task_type name_entity_recognition

python -m data.advanced_partition.util.visualization_heatmap_label \
--partition_name  'niid_label_clients=100_alpha=1.0' \
--client_num 100 \
--partition_file ${DATA_DIR}/partition_files/agnews_partition.h5 \
--data_file ${DATA_DIR}/data_files/agnews_data.h5 \
--task_name agnews\
--figure_path ${DATA_DIR}/advanced_partition/heatmap_figure \
--task_type text_classification 


python -m data.advanced_partition.util.visualization_distplot \
--client_num 100 \
--partition_file ${DATA_DIR}/partition_files/20news_partition.h5\
--data_file ${DATA_DIR}/data_files/20news_data.h5\
--task_name 20news \
--cluster_num 0 \
--figure_path ${DATA_DIR}/advanced_partition/dist_figure\
--task_type text_classification 

python -m data.advanced_partition.util.visualization_distplot \
--client_num 100 \
--partition_file ${DATA_DIR}/partition_files/wikiner_partition.h5\
--data_file ${DATA_DIR}/data_files/wikiner_data.h5\
--task_name wikiner \
--cluster_num 0 \
--figure_path ${DATA_DIR}/advanced_partition/dist_figure\
--task_type name_entity_recognition 

python -m data.advanced_partition.util.visualization_quantity_distplot \
--client_num 100  \
--partition_file ${DATA_DIR}/partition_files/20news_partition.h5\
--data_file ${DATA_DIR}/data_files/20news_data.h5\
--task_name 20news \
--cluster_num 0 \
--figure_path ${DATA_DIR}/advanced_partition/dist_figure\
--task_type text_classification 

python -m data.advanced_partition.util.visualization_quantity_distplot \
--client_num 100 \
--partition_file ${DATA_DIR}/partition_files/wikiner_partition.h5\
--data_file ${DATA_DIR}/data_files/wikiner_data.h5\
--task_name wikiner \
--cluster_num 0 \
--figure_path ${DATA_DIR}/advanced_partition/dist_figure\
--task_type name_entity_recognition 



```
