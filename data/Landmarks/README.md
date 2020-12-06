# Download Google Landmarks datasets (gld160k or gld23k)


You are recommended to download images.zip and data_user_dict.zip file from these two .zip files:
https://drive.google.com/file/d/1QyOdNO0LhjVXcGlREz1hMQ8oyPSZm8vn/view?usp=sharing

https://drive.google.com/file/d/17psh9F7vZs_V60AwX4n86dBHNwfujXQK/view?usp=sharing

The images.zip is 23GB and data_user_dict.zip is 2MB. After downloading, please unzip them into your self-defined data path.



We also provide some other scripts to download. These downloading scripts follow tensorflow_federated, which downloads data from http://storage.googleapis.com/gresearch/federated-vision-datasets/%s.zip. The total size of these files is 500GB. And they do not support Breakpoint retransmission. So it is easy to fail to download them. You can use these scripts as following (Not Recommend):

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
