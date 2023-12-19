#!/usr/bin/env/python
import os
import boto3
from botocore.client import Config


s3 = boto3.resource('s3',
                    endpoint_url='http://127.0.0.1:9000',
                    aws_access_key_id='AKIAUAWARWF4SW36VYXP',
                    aws_secret_access_key='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',
                    config=Config(signature_version='s3v4'),
                    region_name='us-west-1')

# upload a file from local file system './remote_storage.py' to bucket 'songs' with 'remote_storage.py' as the object name.
# s3.create_bucket(Bucket='fedml')
bucket = s3.Bucket('fedml')

if bucket.creation_date:
   print("The bucket exists")
else:
   print("The bucket does not exist")
   s3.create_bucket(Bucket='fedml')


curr_dir = os.path.dirname(os.path.realpath(__file__))
remote_storage_file_path_py = os.path.join(curr_dir, 'remote_storage.py')
s3.Bucket('fedml').upload_file(remote_storage_file_path_py,'remote_storage.xxx')
print("uploaded")

# download the object 'piano.mp3' from the bucket 'songs' and save it to local FS as /tmp/classical.mp3
remote_storage_file_path_yyy = os.path.join(curr_dir, 'remote_storage.yyy')
s3.Bucket('fedml').download_file('remote_storage.xxx', remote_storage_file_path_yyy)
print("downloaded")