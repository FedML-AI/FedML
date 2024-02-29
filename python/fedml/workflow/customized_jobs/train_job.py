import json
import os
import shutil
import traceback
import uuid

import fedml.api
from fedml.workflow.customized_jobs.customized_base_job import CustomizedBaseJob
from fedml.computing.scheduler.comm_utils import sys_utils
from typing import List, Dict, Any
from os.path import expanduser
import base64


class TrainJob(CustomizedBaseJob):
    TRAIN_JOB_INPUTS_CONFIG = "train_job_inputs"
    TRAIN_JOB_INPUTS_ENV = f"FEDML_ENV_{str(TRAIN_JOB_INPUTS_CONFIG).upper()}"
    TRAIN_JOB_OUTPUTS_KEY_PREFIX = "FEDML_TRAIN_JOB_OUTPUTS"
    RUN_API_KEY_CONFIG = "run_api_key"
    RUN_API_KEY_ENV = f"FEDML_ENV_{str(RUN_API_KEY_CONFIG).upper()}"

    def __init__(self, name, job_yaml_absolute_path=None, job_api_key=None):
        super().__init__(name, job_yaml_absolute_path=job_yaml_absolute_path,
                         job_api_key=job_api_key)
        self.in_trainning_params = None
        self.out_model_file = ""
        self.job_yaml_dir = os.path.dirname(self.job_yaml_absolute_path)
        self.job_yaml_absolute_path_for_launch = os.path.join(
            self.job_yaml_dir, f"{str(uuid.uuid4())}.yaml")

    def run(self):
        job_yaml_obj = self.load_yaml_config(self.job_yaml_absolute_path)
        job_yaml_obj[TrainJob.TRAIN_JOB_INPUTS_CONFIG] = TrainJob._base64_encode(self.input_data_dict)
        job_yaml_obj[TrainJob.RUN_API_KEY_CONFIG] = sys_utils.random1(f"FEDML_NEXUS@{self.job_api_key}", "FEDML@88119999GREAT")
        self.generate_yaml_doc(job_yaml_obj, self.job_yaml_absolute_path_for_launch)
        self.job_yaml_absolute_path = self.job_yaml_absolute_path_for_launch

        super().run()

        os.remove(self.job_yaml_absolute_path_for_launch)

    def status(self):
        return super().status()

    def kill(self):
        super().kill()

    @staticmethod
    def get_inputs():
        input_data = os.environ.get(TrainJob.TRAIN_JOB_INPUTS_ENV)
        if input_data is None:
            return None
        else:
            return TrainJob._base64_decode(input_data)

    @staticmethod
    def set_outputs(output_dict: Dict[Any, Any]):
        try:
            output_file = TrainJob._get_output_file()
            with open(output_file, "w") as f:
                f.write(json.dumps(output_dict))
                f.write("\n\n")
            run_id = os.getenv('FEDML_CURRENT_RUN_ID', 0)
            output_name = f"{TrainJob.TRAIN_JOB_OUTPUTS_KEY_PREFIX}_{run_id}"
            current_job_api_key = TrainJob._get_job_api_key()
            random_out = sys_utils.random2(current_job_api_key, "FEDML@88119999GREAT")
            random_list = random_out.split("FEDML_NEXUS@")
            response = fedml.api.upload(data_path=output_file, service='S3', name=output_name,
                                        api_key=random_list[1])
            print(f"upload response: code {response.code}, message {response.message}, data {response.data}")
            shutil.rmtree(os.path.dirname(output_file), ignore_errors=True)
        except Exception as e:
            print(f"exception when setting outputs {traceback.format_exc()}.")

    def get_outputs(self):
        try:
            output_file = self._get_output_file()
            output_dir = os.path.dirname(output_file)
            output_name = f"{TrainJob.TRAIN_JOB_OUTPUTS_KEY_PREFIX}_{self.run_id}"
            response = fedml.api.download(output_name, service='S3', api_key=self.job_api_key,
                                          dest_path=output_dir, show_progress=False)
            downloaded_file_list = os.listdir(output_dir)
            if downloaded_file_list is None or len(downloaded_file_list) <= 0:
                return {}
            downloaded_file = os.path.join(output_dir, downloaded_file_list[0])
            with open(downloaded_file, "r") as f:
                output_data = f.readlines()
                output_json = json.loads(output_data[0])
                self.output_data_dict = output_json
            shutil.rmtree(output_dir, ignore_errors=True)
            print(f"download response: code {response.code}, message {response.message}, data {response.data}")
        except Exception as e:
            print(f"exception when getting outputs {traceback.format_exc()}.")

        return self.output_data_dict

    @staticmethod
    def _get_output_file():
        home_dir = expanduser("~")
        cache_dir = os.path.join(home_dir, ".cache", "fedml_jobs", str(uuid.uuid4()))
        os.makedirs(cache_dir, exist_ok=True)
        output_file = os.path.join(cache_dir, str(uuid.uuid4()))
        return output_file

    @staticmethod
    def _get_job_api_key():
        return os.environ.get(TrainJob.RUN_API_KEY_ENV, "")

    @staticmethod
    def _base64_encode(input: dict) -> str:
        message_bytes = json.dumps(input).encode("ascii")
        base64_bytes = base64.b64encode(message_bytes)
        return base64_bytes.decode("ascii")

    @staticmethod
    def _base64_decode(input: str) -> dict:
        message_bytes = input.encode("ascii")
        base64_bytes = base64.b64decode(message_bytes)
        payload = base64_bytes.decode("ascii")
        return json.loads(payload)



