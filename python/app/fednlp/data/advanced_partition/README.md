# Advanced Partition Method

## BERT-based Clustering 
We first use sentence transformer to compute the embedding of the data and then run kmeans to gets K clusters based on client numbers.


```bash
# overwrite switch stores False the default value is True, 
# if you want to create a new emebdding file add the overwrite switch on otherwise delete the argument the default value is True 
# so that it will automatically use the 3 existing embedding file 
# use overwrite only if you want to create a new embedding file or do not have an exisiting embedding file 
# for example the current avaliable gpu is the first GPU use the export CUDA_VISIBLE_DEVICES=0 \ if the current avaliable 
# GPU is the fourth one use the export CUDA_VISIBLE_DEVICES=3 \

DATA_DIR=~/fednlp_data/

CUDA_VISIBLE_DEVICES=0 \
python -m data.advanced_partition.kmeans  \
    --cluster_number 10 \
    --data_file ${DATA_DIR}/data_files/wikiner_data.h5 \
    --bsz 16 \
    --partition_file ${DATA_DIR}/partition_files/wikiner_partition.h5 \
    --embedding_file ${DATA_DIR}/embedding_files/wikiner_embedding.pkl  \
    --task_type name_entity_recognition \
    --overwrite  
    
CUDA_VISIBLE_DEVICES=0 \
python -m data.advanced_partition.kmeans  \
    --cluster_number 30 \
    --data_file ${DATA_DIR}/data_files/squad_1.1_data.h5 \
    --bsz 16 \
    --partition_file ${DATA_DIR}/partition_files/squad_1.1_partition.h5 \
    --embedding_file ${DATA_DIR}/embedding_files/squad_1.1_embedding.pkl  \
    --task_type reading_comprehension \
    --overwrite

CUDA_VISIBLE_DEVICES=0 \
python -m data.advanced_partition.kmeans  \
    python kmeans_ex.py  \
    --cluster_number 50 \
    --data_file ${DATA_DIR}/data_files/cornell_movie_dialogue_data.h5 \
    --bsz 16 \
    --partition_file ${DATA_DIR}/partition_files/cornell_movie_dialogue_partition.h5 \
    --embedding_file ${DATA_DIR}/embedding_files/cornell_movie_dialogue_embedding.pkl  \
    --task_type seq2seq \
    --overwrite

```

## niid_label_skew\ niid cluster skew

we first use kmeans clustering to classify some datasets in to {10,30,50} clusters and then calculate dirichlet distribution of all labels within each client 

We already provide clusters data for datasets excluding **20news**, **agnews**, **sst2** because they have their own natural classification. 

In the each of the rest partition h5 files, you can access the clustering data by the keyword "**kmeans_%client_number**" based on how many client number you assign in the Kmeans partition and you can also find which data belongs to which cluster under the keyword **kmeans_%client_number/cluster_assignment** . 
you can access the partition data by the keyword **niid_cluster_clients=%client_number_alpha=%alpha** for non text-classification tasks, alpha and client is the value you input. If you would like to create different numbers of clusters you can use the kmeans code we provide above. 

For text_classification data we use their natural label to form the paritition so in **20news**, **agnews**, **sst2**'s partition files, you can access partition by keyword **niid_label_clients=%client_number_alpha=%alpha** for text classification task where alpha is the value you input

### Usage

```bash
DATA_DIR=~/fednlp_data/

python -m data.advanced_partition.niid_label \
--client_number 100 \
--data_file ${DATA_DIR}/data_files/20news_data.h5 \
--partition_file ${DATA_DIR}/partition_files/20news_partition.h5 \
--task_type text_classification \
--skew_type label \
--seed 42 \
--kmeans_num 0  \
--alpha 0.5


python -m data.advanced_partition.niid_label \
--client_number 1000 \
--data_file ${DATA_DIR}/data_files/agnews_data.h5 \
--partition_file ${DATA_DIR}/partition_files/agnews_partition.h5 \
--task_type text_classification \
--skew_type label \
--seed 42 \
--kmeans_num 0  \
--alpha 0.5

python -m data.advanced_partition.niid_label   \
--client_number 30 \
--data_file ${DATA_DIR}/data_files/sst_2_data.h5 \
--partition_file ${DATA_DIR}/partition_files/sst_2_partition.h5 \
--task_type text_classification \
--skew_type label \
--seed 42 \
--kmeans_num 0 \
--alpha 0.5


python -m data.advanced_partition.niid_label   \
--client_number 1000 \
--data_file ${DATA_DIR}/data_files/wikiner_data.h5 \
--partition_file ${DATA_DIR}/partition_files/wikiner_partition.h5 \
--task_type sequence_tagging \
--skew_type label \
--seed 42 \
--kmeans_num 10  \
--alpha 0.5

python -m data.advanced_partition.niid_label   \
--client_number 30 \
--data_file ${DATA_DIR}/data_files/w_nut_data.h5 \
--partition_file ${DATA_DIR}/partition_files/w_nut_partition.h5 \
--task_type sequence_tagging \
--skew_type label \
--seed 42 \
--kmeans_num 10  \
--alpha 0.5


python -m data.advanced_partition.niid_label   \
--client_number 300 \
--data_file ${DATA_DIR}/data_files/squad_1.1_data.h5 \
--partition_file ${DATA_DIR}/partition_files/squad_1.1_partition.h5 \
--task_type reading_comprehension \
--skew_type label \
--seed 42 \
--kmeans_num 30  \
--alpha 0.5
```

## niid_quantity_skew

In this partition method. We propose a partition method that distribute our data based solely on qunatity of data.
Therefore, we assume all the data has only one label so we will use Dirichlet Distribution of 
the quantities. beta will be used to calculate Dirichlet Distribution

``` bash

python -m data.advanced_partition.niid_quantity  \
--client_number 100  \
--data_file ${DATA_DIR}/data_files/20news_data.h5  \
--partition_file ${DATA_DIR}/partition_files/20news_partition.h5 \
--task_type text_classification \
--kmeans_num 0 \
--beta 5


python -m data.advanced_partition.niid_quantity  \
--client_number 1000  \
--data_file ${DATA_DIR}/data_files/agnews_data.h5 \
--partition_file ${DATA_DIR}/partition_files/agnews_partition.h5 \
--task_type text_classification \
--kmeans_num 0  \
--beta 5

python -m data.advanced_partition.niid_quantity  \
--client_number 30 \
--data_file ${DATA_DIR}/data_files/sst_2_data.h5 \
--partition_file ${DATA_DIR}/partition_files/sst_2_partition.h5 \
--task_type text_classification \
--kmeans_num 0 \
--beta 5


python -m data.advanced_partition.niid_quantity  \
--client_number 100 \
--data_file ${DATA_DIR}/data_files/wikiner_data.h5 \
--partition_file ${DATA_DIR}/partition_files/wikiner_partition.h5 \
--task_type name_entity_recognition \
--kmeans_num 10 \
--beta 5

python -m data.advanced_partition.niid_quantity \
--client_number 30 \
--data_file ${DATA_DIR}/data_files/w_nut_data.h5 \
--partition_file ${DATA_DIR}/partition_files/w_nut_partition.h5\
--task_type name_entity_recognition --kmeans_num 10  \
--beta 5


python -m data.advanced_partition.niid_quantity   \
--client_number 300 \
--data_file ${DATA_DIR}/data_files/squad_1.1_data.h5 \
--partition_file ${DATA_DIR}/partition_files/squad_1.1_partition.h5 \
--task_type reading_comprehension \
--kmeans_num 30  \
--beta 5



```


