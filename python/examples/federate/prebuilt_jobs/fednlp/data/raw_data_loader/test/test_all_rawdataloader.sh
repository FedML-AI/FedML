#!/bin/bash
declare -a data_names=("20news" "agnews" "cnn_dailymail" "cornell_movie_dialogue" "semeval_2010_task8" "sentiment140" "squad_1.1" "sst_2" "w_nut" "wikiner" "wmt_zh-en" "wmt_cs-en" "wmt_ru-en" "wmt_de-en")

declare DATA_BASE_DIR="../../../data/download_scripts"
declare -a data_dir_paths=("${DATA_BASE_DIR}/text_classification/20Newsgroups" "${DATA_BASE_DIR}/text_classification/AGNews" 
    "${DATA_BASE_DIR}/seq2seq/CNN_Dailymail" "${DATA_BASE_DIR}/seq2seq/CornellMovieDialogue/cornell_movie_dialogs_corpus" 
    "${DATA_BASE_DIR}/text_classification/SemEval2010Task8/SemEval2010_task8_all_data" "${DATA_BASE_DIR}/text_classification/Sentiment140" 
    "${DATA_BASE_DIR}/span_extraction/SQuAD_1.1" "${DATA_BASE_DIR}/text_classification/SST-2/trees" "${DATA_BASE_DIR}/sequence_tagging/W-NUT2017" 
    "${DATA_BASE_DIR}/sequence_tagging/wikiner" 
    "${DATA_BASE_DIR}/seq2seq/WMT/training-parallel-nc-v13/news-commentary-v13.zh-en.zh,${DATA_BASE_DIR}/seq2seq/WMT/training-parallel-nc-v13/news-commentary-v13.zh-en.en" 
    "${DATA_BASE_DIR}/seq2seq/WMT/training-parallel-nc-v13/news-commentary-v13.cs-en.cs,${DATA_BASE_DIR}/seq2seq/WMT/training-parallel-nc-v13/news-commentary-v13.cs-en.en" 
    "${DATA_BASE_DIR}/seq2seq/WMT/training-parallel-nc-v13/news-commentary-v13.ru-en.ru,${DATA_BASE_DIR}/seq2seq/WMT/training-parallel-nc-v13/news-commentary-v13.ru-en.en" 
    "${DATA_BASE_DIR}/seq2seq/WMT/training-parallel-nc-v13/news-commentary-v13.de-en.de,${DATA_BASE_DIR}/seq2seq/WMT/training-parallel-nc-v13/news-commentary-v13.de-en.en")


for i in "${!data_names[@]}"
do
	python ./test_rawdataloader.py --dataset ${data_names[$i]} --data_dir ${data_dir_paths[$i]} --h5_file_path ./${data_names[$i]}_data.h5
done

