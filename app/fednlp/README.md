**FedNLP INTRO**

We provide 4 different NLP applications namely Text Classification, Sequence Tagging, Span Extraction and Sequence2Sequence. We provide examples for each application and also provide steps on how to run each application below. We have provided download scripts for 12 different datasets across these 4 applications.

**TEXT CLASSIFICATION**

Read data/README.md for more details of datasets available

Adjust the hyperparameters in fednlp/text_classification/config/fedml_config.yaml

To run text classification using MPI simulator follow the following steps:

1. cd ../
2. bash fednlp/data/download_data.sh
3. bash fednlp/data/download_partition.sh
4. bash fednlp/text_classification/run_step_by_step_example.sh


**SEQ TAGGING**

Read data/README.md for more details of datasets available

Adjust the hyperparameters in fednlp/seq_tagging/config/fedml_config.yaml

To run sequence tagging using MPI simulator follow the following steps:

1. cd ../
2. bash fednlp/data/download_data.sh
3. bash fednlp/data/download_partition.sh
4. bash fednlp/seq_tagging/run_step_by_step_example.sh


**SPAN EXTRACTION**

Adjust the hyperparameters in fednlp/span_extraction/config/fedml_config.yaml and make sure data file paths are correct

To run span extraction on SQuAD1.1 dataset using MPI simulator follow the following steps:

1. cd ../
2. bash fednlp/data/download_scripts/span_extraction/SQuAD_1.1/download.sh
3. python fednlp/data/raw_data_loader/test/test_rawdataloader.py --dataset squad_1.1 --data_dir ./ --h5_file_path ./squad1.1_data.h5
4. bash fednlp/data/download_partition.sh
5. bash fednlp/span_extraction/run_step_by_step_example.sh


**SEQ2SEQ**

Read data/README.md for more details of datasets available

Adjust the hyperparameters in fednlp/seq2seq/config/fedml_config.yaml

To run seq2seq using MPI simulator follow the following steps:

1. cd ../
2. bash fednlp/data/download_data.sh
3. bash fednlp/data/download_partition.sh
4. bash fednlp/seq_tagging/run_step_by_step_example.sh


We have provided examples of trainers in each example. For running custom trainers feel free to follow the folder {application_name}/trainer/ and write your own custom trainer. To include this trainer please follow the create_model function in the python executable in the folder {application_name}/ and replace the current trainer with your own trainer. Every trainer should inherit the Client Trainer class and should contain a train and a test function.


We have provided examples with BERT and DistilBert for text classification, seq tagging and span extraction and BART for Seq2Seq. For using any other model from Huggingface Transformers please look at the create_model function in the python executable in the folder {application_name}/. Also please ensure that you are using the correct tokenizer in {application_name}/data/data_loader.py 


* Here {application_name} refers to any one of text_classification, span_extraction, seq_tagging or seq2seq.
