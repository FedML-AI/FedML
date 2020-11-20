## DataLoader

In fednlp, we design two kinds of data loaders for users to load the dataset since some users care about data formats,
data distribution and data features but others are not.

### RawDataLoader
RawDataLoader load and process dataset from scratch. Users can specify the dataset path in constructor and call the 
data_loader function to load the data. The return format will be a dictionary which contains raw data and data attributes.
Users can exploit the data attributes or data itself for further partition.

### ClientDataLoader
For those users who do not care about raw data, we pre-generate a data pickle file and a partition pickle file for each 
dataset. In this case, users can utilize ClientDataLoader to load the entire data or partitioned data by assigning client_idx
with reasonable value in constructor. Also, users can specify the partition method in order to obtaining data with 
different partition. For more details, please checkout the code directly.

The figure below illustrates the workflow.
![avatar](docs/images/data_loader_work_flow.jpg)


### Test
You can try the following commands and checkout /test/run_fednlp_dataloader.sh as well as test_dataloader.py for more
details.

#### Test Scripts
``` 
## 20news
sh run_fednlp_dataloader.sh 20news ../../../../data/fednlp/text_classification/20Newsgroups/20news-18828 uniform 32 100

## AGNews
sh run_fednlp_dataloader.sh agnews ../../../../data/fednlp/text_classification/AGNews uniform 32 100

## CNN_Dailymail
sh run_fednlp_dataloader.sh cnn_dailymail ../../../../data/fednlp/seq2seq/CNN_Dailymail uniform 32 100

## CornellMovieDialogue
sh run_fednlp_dataloader.sh cornell_movie_dialogue ../../../../data/fednlp/seq2seq/CornellMovieDialogue/cornell_movie_dialogs_corpus uniform 32 100

## SemEval2010Task8
sh run_fednlp_dataloader.sh semeval_2010_task8 ../../../../data/fednlp/text_classification/SemEval2010Task8/SemEval2010_task8_all_data uniform 32 100

## Sentiment140
sh run_fednlp_dataloader.sh sentiment_140 ../../../../data/fednlp/text_classification/Sentiment140 uniform 32 100

## SQuAD_1.1
sh run_fednlp_dataloader.sh squad_1.1 ../../../../data/fednlp/span_extraction/SQuAD_1.1 uniform 32 100

## SST-2
sh run_fednlp_dataloader.sh sst_2 ../../../../data/fednlp/text_classification/SST-2/stanfordSentimentTreebank uniform 32 100

## W_NUT
sh run_fednlp_dataloader.sh w_nut ../../../../data/fednlp/sequence_tagging/W-NUT2017/data uniform 32 100

## wikigold
sh run_fednlp_dataloader.sh wikigold ../../../../data/fednlp/sequence_tagging/wikigold/wikigold/CONLL-format/data/wikigold.conll.txt uniform 32 100

## WMT
sh run_fednlp_dataloader.sh wmt ../../../../data/fednlp/seq2seq/WMT/training-parallel-nc-v13/news-commentary-v13.zh-en.zh,../../../../data/fednlp/seq2seq/WMT/training-parallel-nc-v13/news-commentary-v13.zh-en.en uniform 32 100
```