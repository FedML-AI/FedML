import io
import json
import logging
import os
import pickle
import sys
import time
import uuid
from os.path import expanduser

import boto3
# for multi-processing, we need to create a global variable for AWS S3 client:
# https://www.pythonforthelab.com/blog/differences-between-multiprocessing-windows-and-linux/
# https://stackoverflow.com/questions/72313845/multiprocessing-picklingerror-cant-pickle-class-botocore-client-s3-attr
import dill
import torch
import tqdm
import yaml
from fedml.core.distributed.communication.s3.utils import load_params_from_tf, process_state_dict
from fedml.core.mlops.mlops_profiler_event import MLOpsProfilerEvent
from torch import nn

aws_s3_client = None
aws_s3_resource = None


class S3Storage:
    def __init__(self, s3_config_path):
        self.bucket_name = None
        self.cn_region_name = None
        self.cn_s3_sak = None
        self.cn_s3_aki = None
        self.set_config_from_file(s3_config_path)
        self.set_config_from_objects(s3_config_path)
        global aws_s3_client
        aws_s3_client = boto3.client(
            "s3",
            region_name=self.cn_region_name,
            aws_access_key_id=self.cn_s3_aki,
            aws_secret_access_key=self.cn_s3_sak,
        )

        global aws_s3_resource
        aws_s3_resource = boto3.resource(
            "s3",
            region_name=self.cn_region_name,
            aws_access_key_id=self.cn_s3_aki,
            aws_secret_access_key=self.cn_s3_sak,
        )

    def write_model(self, message_key, model):
        global aws_s3_client
        pickle_dump_start_time = time.time()
        MLOpsProfilerEvent.log_to_wandb(
            {"PickleDumpsTime": time.time() - pickle_dump_start_time}
        )
        # for python clients
        model_pkl = pickle.dumps(model)
        model_to_send = model_pkl  # bytes object
        s3_upload_start_time = time.time()

        model_file_size = len(model_to_send)
        model_file_transfered = 0
        prev_progress = 0
        def upload_model_progress(bytes_transferred):
            nonlocal model_file_transfered
            nonlocal model_file_size
            nonlocal prev_progress  # since the callback is stateless, we need to keep the previous progress
            model_file_transfered += bytes_transferred
            uploaded_kb = format(model_file_transfered / 1024, '.2f')
            progress = (model_file_transfered / model_file_size * 100) if model_file_size != 0 else 0
            progress_format_int = int(progress)
            # print the process every 5%
            if progress_format_int % 5 == 0 and progress_format_int != prev_progress:
                logging.info("model uploaded to S3 size {} KB, progress {}%".format(uploaded_kb, progress_format_int))
                prev_progress = progress_format_int
        
        aws_s3_client.upload_fileobj(
            Fileobj=io.BytesIO(model_to_send), Bucket=self.bucket_name, Key=message_key,
            Callback=upload_model_progress,
        )
        MLOpsProfilerEvent.log_to_wandb(
            {"Comm/send_delay": time.time() - s3_upload_start_time}
        )
        model_url = aws_s3_client.generate_presigned_url(
            "get_object",
            ExpiresIn=60 * 60 * 24 * 5,
            Params={"Bucket": self.bucket_name, "Key": message_key},
        )
        return model_url

    def write_model_net(self, message_key, model, dummy_input_tensor, local_model_cache_path):
        global aws_s3_client
        pickle_dump_start_time = time.time()
        MLOpsProfilerEvent.log_to_wandb(
            {"PickleDumpsTime": time.time() - pickle_dump_start_time}
        )

        if not os.path.exists(local_model_cache_path):
            os.makedirs(local_model_cache_path, exist_ok=True)
        write_model_path = os.path.join(local_model_cache_path, message_key)
        try:
            model.eval()
            jit_model = torch.jit.trace(model, dummy_input_tensor)
            jit_model.save(write_model_path)
        except Exception as e:
            logging.info("jit.save failed")
            torch.save(model, write_model_path, pickle_module=dill)

        s3_upload_start_time = time.time()

        with open(write_model_path, 'rb') as f:
            model_to_send = io.BytesIO(f.read())

        model_to_send.seek(0, 2)
        net_file_size = model_to_send.tell()
        model_to_send.seek(0, 0)
        net_file_transfered = 0
        prev_progress = 0
        def upload_model_net_progress(bytes_transferred):
            nonlocal net_file_transfered
            nonlocal net_file_size
            nonlocal prev_progress  # since the callback is stateless, we need to keep the previous progress
            net_file_transfered += bytes_transferred
            uploaded_kb = format(net_file_transfered / 1024, '.2f')
            progress = (net_file_transfered / net_file_size * 100) if net_file_size != 0 else 0
            progress_format_int = int(progress)
            # print the process every 5%
            if progress_format_int % 5 == 0 and progress_format_int != prev_progress:
                logging.info("model net uploaded to S3 size {} KB, progress {}%".format(uploaded_kb, progress_format_int))
                prev_progress = progress_format_int
        aws_s3_client.upload_fileobj(
            Fileobj=model_to_send, Bucket=self.bucket_name, Key=message_key,
            Callback= upload_model_net_progress,
        )
        MLOpsProfilerEvent.log_to_wandb(
            {"Comm/send_delay": time.time() - s3_upload_start_time}
        )
        model_url = aws_s3_client.generate_presigned_url(
            "get_object",
            ExpiresIn=60 * 60 * 24 * 5,
            Params={"Bucket": self.bucket_name, "Key": message_key},
        )
        return model_url

    def write_model_input(self, message_key, input_size, input_type, local_model_cache_path):
        global aws_s3_client

        if not os.path.exists(local_model_cache_path):
            try:
                os.makedirs(local_model_cache_path)
            except Exception as e:
                pass
        model_input_path = os.path.join(local_model_cache_path, message_key)
        model_input_dict = {"input_size": input_size, "input_type": input_type}
        with open(model_input_path, "w") as f:
            json.dump(model_input_dict, f)

        with open(model_input_path, 'rb') as f:
            aws_s3_client.upload_fileobj(f, Bucket=self.bucket_name, Key=message_key)

        model_input_url = aws_s3_client.generate_presigned_url("get_object",
                                                               ExpiresIn=60 * 60 * 24 * 5,
                                                               Params={"Bucket": self.bucket_name, "Key": message_key})
        return model_input_url

    def write_model_web(self, message_key, model):
        global aws_s3_client
        pickle_dump_start_time = time.time()
        MLOpsProfilerEvent.log_to_wandb(
            {"PickleDumpsTime": time.time() - pickle_dump_start_time}
        )
        # for javascript clients
        state_dict = model  # type: OrderedDict
        model_json = process_state_dict(state_dict)
        model_to_send = json.dumps(model_json)
        s3_upload_start_time = time.time()
        aws_s3_client.put_object(
            Body=model_to_send, Bucket=self.bucket_name, Key=message_key, ACL="public-read",
        )
        MLOpsProfilerEvent.log_to_wandb(
            {"Comm/send_delay": time.time() - s3_upload_start_time}
        )
        model_url = aws_s3_client.generate_presigned_url(
            "get_object",
            ExpiresIn=60 * 60 * 24 * 5,
            Params={"Bucket": self.bucket_name, "Key": message_key},
        )
        return model_url

    def read_model(self, message_key):
        global aws_s3_client
        message_handler_start_time = time.time()

        kwargs = {"Bucket": self.bucket_name, "Key": message_key}
        object_size = aws_s3_client.head_object(**kwargs)["ContentLength"]
        cache_dir = os.path.join(expanduser("~"), "fedml_cache")
        if not os.path.exists(cache_dir):
            try:
                os.makedirs(cache_dir)
            except Exception as e:
                pass
        temp_base_file_path = os.path.join(cache_dir, str(os.getpid()) + "@" + str(uuid.uuid4()))
        if not os.path.exists(temp_base_file_path):
            try:
                os.makedirs(temp_base_file_path)
            except Exception as e:
                pass

        temp_file_path = temp_base_file_path + "/" + str(message_key)
        logging.info("temp_file_path = {}".format(temp_file_path))
        model_file_transfered = 0
        prev_progress = 0
        def read_model_progress(bytes_transferred):
            nonlocal model_file_transfered
            nonlocal object_size
            nonlocal prev_progress
            model_file_transfered += bytes_transferred
            readed_kb = format(model_file_transfered / 1024, '.2f')
            progress = (model_file_transfered / object_size * 100) if object_size != 0 else 0
            progress_format_int = int(progress)
            # print the process every 5%
            if progress_format_int % 5 == 0 and progress_format_int != prev_progress:
                logging.info("model readed from S3 size {} KB, progress {}%".format(readed_kb, progress_format_int))
                prev_progress = progress_format_int

        with open(temp_file_path, 'wb') as f:
            aws_s3_client.download_fileobj(Bucket=self.bucket_name, Key=message_key, Fileobj=f,
                                            Callback=read_model_progress)
        MLOpsProfilerEvent.log_to_wandb(
            {"Comm/recieve_delay_s3": time.time() - message_handler_start_time}
        )

        unpickle_start_time = time.time()
        with open(temp_file_path, 'rb') as model_pkl_file:
            model = pickle.load(model_pkl_file)
        os.remove(temp_file_path)
        os.rmdir(temp_base_file_path)
        MLOpsProfilerEvent.log_to_wandb(
            {"UnpickleTime": time.time() - unpickle_start_time}
        )
        return model

    def read_model_net(self, message_key, local_model_cache_path):
        global aws_s3_client
        message_handler_start_time = time.time()

        kwargs = {"Bucket": self.bucket_name, "Key": message_key}
        object_size = aws_s3_client.head_object(**kwargs)["ContentLength"]
        temp_base_file_path = local_model_cache_path
        if not os.path.exists(temp_base_file_path):
            try:
                os.makedirs(temp_base_file_path)
            except Exception as e:
                pass
        temp_file_path = os.path.join(temp_base_file_path, str(message_key))
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)
        logging.info("temp_file_path = {}".format(temp_file_path))
        model_file_transfered = 0
        prev_progress = 0
        def read_model_net_progress(bytes_transferred):
            nonlocal model_file_transfered
            nonlocal object_size
            nonlocal prev_progress  # since the callback is stateless, we need to keep the previous progress
            model_file_transfered += bytes_transferred
            readed_kb = format(model_file_transfered / 1024, '.2f')
            progress = (model_file_transfered / object_size * 100) if object_size != 0 else 0
            progress_format_int = int(progress)
            # print the process every 5%
            if progress_format_int % 5 == 0 and progress_format_int != prev_progress:
                logging.info("model net readed from S3 size {} KB, progress {}%".format(readed_kb, progress_format_int))
                prev_progress = progress_format_int
        with open(temp_file_path, 'wb') as f:
            aws_s3_client.download_fileobj(Bucket=self.bucket_name, Key=message_key, Fileobj=f,
                                            Callback=read_model_net_progress)
        MLOpsProfilerEvent.log_to_wandb(
            {"Comm/recieve_delay_s3": time.time() - message_handler_start_time}
        )

        unpickle_start_time = time.time()
        try:
            model = torch.jit.load(temp_file_path)
        except Exception as e:
            logging.info("jit.load failed")
            model = torch.load(temp_file_path, pickle_module=dill)
        os.remove(temp_file_path)
        MLOpsProfilerEvent.log_to_wandb(
            {"UnpickleTime": time.time() - unpickle_start_time}
        )
        return model

    def read_model_input(self, message_key, local_model_cache_path):
        global aws_s3_client

        temp_base_file_path = local_model_cache_path
        if not os.path.exists(temp_base_file_path):
            try:
                os.makedirs(temp_base_file_path)
            except Exception as e:
                pass
        temp_file_path = os.path.join(temp_base_file_path, str(message_key))
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)
        logging.info("temp_file_path = {}".format(temp_file_path))
        with open(temp_file_path, 'wb') as f:
            aws_s3_client.download_fileobj(Bucket=self.bucket_name, Key=message_key, Fileobj=f)

        with open(temp_file_path, 'r') as f:
            model_input_dict = json.load(f)

        input_size = model_input_dict.get("input_size", None)
        input_type = model_input_dict.get("input_type", None)

        return input_size, input_type

    # TODO: added python torch model to align the Tensorflow parameters from browser
    def read_model_web(self, message_key, py_model: nn.Module):
        global aws_s3_client
        message_handler_start_time = time.time()
        obj = aws_s3_client.get_object(Bucket=self.bucket_name, Key=message_key)
        model_json = obj["Body"].read()
        if type(model_json) == list:
            model = load_params_from_tf(py_model, model_json)
        else:
            model = py_model.state_dict()

        MLOpsProfilerEvent.log_to_wandb(
            {"Comm/recieve_delay_s3": time.time() - message_handler_start_time}
        )
        unpickle_start_time = time.time()
        # model = pickle.loads(model_pkl)
        MLOpsProfilerEvent.log_to_wandb(
            {"UnpickleTime": time.time() - unpickle_start_time}
        )
        return model

    # def write_model(self, message_key, model):
    #     with tempfile.TemporaryFile() as fp:
    #         joblib.dump(model, fp)
    #         fp.seek(0)
    #         global aws_s3_client
    #         aws_s3_client.put_object(
    #             Body=fp.read(),
    #             Bucket=self.bucket_name,
    #             Key=message_key,
    #             ACL="public-read",
    #         )
    #         model_url = aws_s3_client.generate_presigned_url(
    #             "get_object",
    #             ExpiresIn=60 * 60 * 24 * 5,
    #             Params={"Bucket": self.bucket_name, "Key": message_key},
    #         )
    #         return model_url

    # def read_model(self, message_key):
    #     with tempfile.TemporaryFile() as fp:
    #         global aws_s3_client
    #         aws_s3_client.download_fileobj(
    #             Fileobj=fp, Bucket=self.bucket_name, Key=message_key
    #         )
    #         fp.seek(0)
    #         try:
    #             model = joblib.load(fp)
    #         except Exception as e:
    #             print("Exception " + str(e))
    #     return model

    def upload_file(self, src_local_path, message_key):
        """
        upload file
        :param src_local_path:
        :param message_key:
        :return:
        """
        try:
            with open(src_local_path, "rb") as f:
                global aws_s3_client
                aws_s3_client.upload_fileobj(
                    f, self.bucket_name, message_key, ExtraArgs={"ACL": "public-read"}
                )

            model_url = aws_s3_client.generate_presigned_url(
                "get_object",
                ExpiresIn=60 * 60 * 24 * 5,
                Params={"Bucket": self.bucket_name, "Key": message_key},
            )
            return model_url
        except Exception as e:
            logging.error(
                f"Upload data failed. | src: {src_local_path} | dest: {message_key} | Exception: {e}"
            )
            return None
        logging.info(
            f"Uploading file successful. | src: {src_local_path} | dest: {dest_s3_path}"
        )
        return None

    def download_file(self, message_key, path_local):
        """
        download file
        :param message_key: s3 key
        :param path_local: local path
        :return:
        """
        retry = 0
        while retry < 3:
            logging.info(
                f"Start downloading files. | message key: {message_key} | path_local: {path_local}"
            )
            try:
                global aws_s3_client
                aws_s3_client.download_file(self.bucket_name, message_key, path_local)
                file_size = os.path.getsize(path_local)
                logging.info(
                    f"Downloading completed. | size: {round(file_size / 1048576, 2)} MB"
                )
                break
            except Exception as e:
                logging.error(f"Download zip failed. | Exception: {e}")
                retry += 1
        if retry >= 3:
            logging.error(f"Download zip failed after max retry.")

    def upload_file_with_progress(self, src_local_path, dest_s3_path,
                                  out_progress_to_err=True, progress_desc=None):
        """
        upload file
        :param out_progress_to_err:
        :param progress_desc:
        :param src_local_path:
        :param dest_s3_path:
        :return:
        """
        file_uploaded_url = ""
        progress_desc_text = "Uploading Package to AWS S3"
        if progress_desc is not None:
            progress_desc_text = progress_desc
        try:
            with open(src_local_path, "rb") as f:
                global aws_s3_client
                old_file_position = f.tell()
                f.seek(0, os.SEEK_END)
                file_size = f.tell()
                f.seek(old_file_position, os.SEEK_SET)
                with tqdm.tqdm(total=file_size, unit="B", unit_scale=True,
                               file=sys.stderr if out_progress_to_err else sys.stdout,
                               desc=progress_desc_text) as pbar:
                    aws_s3_client.upload_fileobj(
                        f, self.bucket_name, dest_s3_path, ExtraArgs={"ACL": "public-read"},
                        Callback=lambda bytes_transferred: pbar.update(bytes_transferred),
                    )

                file_uploaded_url = aws_s3_client.generate_presigned_url(
                    "get_object",
                    ExpiresIn=60 * 60 * 24 * 365,
                    Params={"Bucket": self.bucket_name, "Key": dest_s3_path},
                )
        except Exception as e:
            logging.error(
                f"Upload data failed. | src: {src_local_path} | dest: {dest_s3_path} | Exception: {e}"
            )
            return file_uploaded_url
        logging.info(
            f"Uploading file successful. | src: {src_local_path} | dest: {dest_s3_path}"
        )
        return file_uploaded_url

    def download_file_with_progress(self, path_s3, path_local,
                                    out_progress_to_err=True, progress_desc=None):
        """
        download file
        :param out_progress_to_err:
        :param progress_desc:
        :param path_s3: s3 key
        :param path_local: local path
        :return:
        """
        retry = 0
        progress_desc_text = "Downloading Package from AWS S3"
        if progress_desc is not None:
            progress_desc_text = progress_desc
        while retry < 3:
            logging.info(
                f"Start downloading files. | path_s3: {path_s3} | path_local: {path_local}"
            )
            try:
                global aws_s3_client
                kwargs = {"Bucket": self.bucket_name, "Key": path_s3}
                object_size = aws_s3_client.head_object(**kwargs)["ContentLength"]
                with tqdm.tqdm(total=object_size, unit="B", unit_scale=True,
                               file=sys.stderr if out_progress_to_err else sys.stdout,
                               desc=progress_desc_text) as pbar:
                    with open(path_local, 'wb') as f:
                        aws_s3_client.download_fileobj(Bucket=self.bucket_name, Key=path_s3, Fileobj=f,
                                                       Callback=lambda bytes_transferred: pbar.update(
                                                           bytes_transferred), )
                break
            except Exception as e:
                logging.error(f"Download zip failed. | Exception: {e}")
                retry += 1
        if retry >= 3:
            logging.error(f"Download zip failed after max retry.")

    def test_s3_base_cmds(self, message_key, message_body):
        """
        test_s3_base_cmds
        :param file_key: s3 message key
        :param file_key: s3 message body
        :return:
        """
        retry = 0
        while retry < 3:
            try:
                global aws_s3_client
                message_pkl = pickle.dumps(message_body)
                aws_s3_client.put_object(
                    Body=message_pkl, Bucket=self.bucket_name, Key=message_key, ACL="public-read",
                )
                obj = aws_s3_client.get_object(Bucket=self.bucket_name, Key=message_key)
                message_pkl_downloaded = obj["Body"].read()
                message_downloaded = pickle.loads(message_pkl_downloaded)
                if str(message_body) == str(message_downloaded):
                    break
                retry += 1
            except Exception as e:
                raise Exception(
                    "S3 base commands test failed at retry count {}, exception: {}".format(str(retry), str(e)))
                retry += 1
        if retry >= 3:
            raise Exception(f"S3 base commands test failed after max retry.")

        return True

    def delete_s3_zip(self, path_s3):
        """
        delete s3 object
        :param path_s3: s3 key
        :return:
        """
        global aws_s3_client
        aws_s3_client.delete_object(Bucket=self.bucket_name, Key=path_s3)
        logging.info(f"Delete s3 file Successful. | path_s3 = {path_s3}")

    def set_config_from_file(self, config_file_path):
        try:
            with open(config_file_path, "r") as f:
                config = yaml.load(f, Loader=yaml.FullLoader)
                self.cn_s3_aki = config["CN_S3_AKI"]
                self.cn_s3_sak = config["CN_S3_SAK"]
                self.cn_region_name = config["CN_REGION_NAME"]
                self.bucket_name = config["BUCKET_NAME"]
        except Exception as e:
            pass

    def set_config_from_objects(self, s3_config):
        self.cn_s3_aki = s3_config["CN_S3_AKI"]
        self.cn_s3_sak = s3_config["CN_S3_SAK"]
        self.cn_region_name = s3_config["CN_REGION_NAME"]
        self.bucket_name = s3_config["BUCKET_NAME"]
