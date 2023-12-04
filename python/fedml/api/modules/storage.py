import boto3
import os
import sys
import tqdm
import logging
import shutil
from fedml.api.modules.utils import authenticate


def _upload_file(archive_path, api_key, show_progress=True, out_progress_to_err=True, progress_desc = None) -> str:
    s3 = boto3.client(service_name='s3',
                      endpoint_url='https://03aa47c68e20656e11ca9e0765c6bc1f.r2.cloudflarestorage.com',
                      aws_access_key_id='44b582b7fe45ec884ec011d34f1da78d',
                      aws_secret_access_key='41ea9ff0536ddf35daf5963da552a6c576c3a66e2359dffe148cf3162d8b2de6',
                      region_name="auto",
                      )
    bucket = "dataset"
    dest_r2_path = os.path.join(api_key, os.path.basename(archive_path))
    file_uploaded_url = ""
    progress_desc_text = "Uploading Package to R2"
    if progress_desc is not None:
        progress_desc_text = progress_desc
    try:
        with open(archive_path, "rb") as f:
            old_file_position = f.tell()
            f.seek(0, os.SEEK_END)
            file_size = f.tell()
            f.seek(old_file_position, os.SEEK_SET)
            if show_progress:
                with tqdm.tqdm(total=file_size, unit="B", unit_scale=True,
                               file=sys.stderr if out_progress_to_err else sys.stdout,
                               desc=progress_desc_text) as pbar:
                    s3.upload_fileobj(f, bucket, dest_r2_path, ExtraArgs={"ACL": "public-read"},
                                      Callback=lambda bytes_transferred: pbar.update(bytes_transferred))
            else:
                s3.upload_fileobj(f, bucket, dest_r2_path, ExtraArgs={"ACL": "public-read"})
            file_uploaded_url = s3.generate_presigned_url(
                "get_object",
                ExpiresIn=60 * 60 * 24 * 7,
                Params={"Bucket": bucket, "Key": dest_r2_path},
            )
    except Exception as e:
        logging.error(
            f"Upload data2 failed. | path: {archive_path} | Exception: {e}"
        )
        return file_uploaded_url
    return file_uploaded_url


def upload(data_path, api_key, name, show_progress, out_progress_to_err, progress_desc) -> str:
    authenticate(api_key)
    src_local_path = os.path.abspath(data_path)
    root_dir = os.path.dirname(src_local_path)
    file_name = os.path.basename(src_local_path) if name is None else name
    archive_name = file_name + '.zip'
    archive_path = os.path.join(root_dir, archive_name)

    try:
        # Create a zip archive
        shutil.make_archive(archive_path[:-4], 'zip', root_dir=root_dir, base_dir=os.path.basename(src_local_path))

        # Now you can use the archive_path for further processing or uploading
        print(f"Successfully archived and uploaded dataset to: {archive_path}")
    except Exception as e:
        print(f"Error archiving dataset: {e}")
    file_uploaded_url = _upload_file(archive_path=archive_path, api_key=api_key, show_progress=show_progress,
                                     out_progress_to_err=out_progress_to_err, progress_desc=progress_desc)
    return file_uploaded_url


def download(data_name, api_key) -> bool:
    authenticate(api_key)
    s3 = boto3.client(service_name='s3',
                      endpoint_url='https://03aa47c68e20656e11ca9e0765c6bc1f.r2.cloudflarestorage.com',
                      aws_access_key_id='44b582b7fe45ec884ec011d34f1da78d',
                      aws_secret_access_key='41ea9ff0536ddf35daf5963da552a6c576c3a66e2359dffe148cf3162d8b2de6',
                      region_name="auto",
                      )
    bucket = "dataset"
    zip_file_name = data_name + ".zip"
    key = os.path.join(api_key, zip_file_name)
    try:
        object = s3.get_object(Bucket=bucket, Key=key)
        data = object['Body'].read()
        dest_folder = os.path.abspath(zip_file_name)
        with open(dest_folder, 'wb') as f:
            f.write(data)
        shutil.unpack_archive(dest_folder, data_name)
    except Exception as e:
        logging.error(
            f"Failed to downlod given file. | file_name: {data_name} | Exception: {e}"
        )
        return False
    return True
