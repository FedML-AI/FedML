# Centralized Training


## LSTM-based 

**TODO**

## Transformer-based 

```bash 
declare -a datasets=("20news" "agnews" "semeval_2010_task8" "sentiment140" "sst_2")
for DATA_NAME in "${datasets[@]}"
do
  CUDA_VISIBLE_DEVICES=1 \
  python -m experiments.centralized.fed_transformer_exps.text_classification \
      --dataset_name ${DATA_NAME} \
      --data_file data/data_loaders/${DATA_NAME}_data_loader.pkl \
      --partition_file data/partition/${DATA_NAME}_partition.pkl \
      --partition_method uniform \
      --model_type distilbert \
      --model_name distilbert-base-uncased \
      --do_lower_case True \
      --train_batch_size 32 \
      --eval_batch_size 32 \
      --max_seq_length 256 \
      --learning_rate 1e-5 \
      --num_train_epochs 5 \
      --output_dir /tmp/${DATA_NAME}_fed/ \
      --n_gpu 1 --fp16
done


```