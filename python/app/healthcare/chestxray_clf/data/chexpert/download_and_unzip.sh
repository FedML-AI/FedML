mkdir $HOME/fedml_data
wget http://download.cs.stanford.edu/deep/CheXpert-v1.0-small.zip -P $HOME/fedml_data
unzip $HOME/fedml_data/CheXpert-v1.0-small.zip -d $HOME/fedml_data/
rm $HOME/fedml_data/CheXpert-v1.0-small.zip
