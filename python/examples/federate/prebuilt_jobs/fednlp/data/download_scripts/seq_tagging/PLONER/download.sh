declare -a data_names=("connl03" "onbc" "onmz" "onnw" "onwb" "wnut16")

for data_name in "${data_names[@]}"
do
    rm ${data_name}_*.txt
    for suffix in train dev test
    do
	    wget https://raw.githubusercontent.com/pfliu-nlp/Named-Entity-Recognition-NER-Papers/master/ner_dataset/PLONER/${data_name}_${suffix}.txt
    done
done
