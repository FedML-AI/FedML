# Download Google Landmarks datasets (gld160k or gld23k)

You are recommended to download images.zip and data_user_dict.zip file from these two .zip files at AWS S3:
```
sh download_from_aws_s3.sh
unzip data_user_dict.zip
unzip images.zip
```

The images.zip is 23GB and data_user_dict.zip is 2MB. After downloading and upzip, please the data related files will look as follows in the current folder.
```
images/*
data_user_dict/gld23k_user_dict_train.csv
data_user_dict/gld23k_user_dict_test.csv
data_user_dict/gld160k_user_dict_train.csv
data_user_dict/gld160k_user_dict_test.csv
```


We also provide some other scripts to download. These downloading scripts follow tensorflow_federated, which downloads data from http://storage.googleapis.com/gresearch/federated-vision-datasets/%s.zip. The total size of these files is 500GB. And they do not support Breakpoint retransmission. So it is easy to fail to download them. You can use these scripts as following 
(Not Recommended because Google script will down 500G files and extract 24G for federated learning):

If you do not have installed tensorflow, you can run 
```
python download_without_tff.py

```



If you have installed tensorflow but not tensorflow_federated, you can run
```
python download_without_tf.py

```



If you have installed tensorflow_federated, you can run
```
python download_with_tff.py

```
