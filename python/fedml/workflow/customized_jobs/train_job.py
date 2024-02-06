import json
import os
import uuid

import fedml.api
from fedml.workflow.customized_jobs.customized_base_job import CustomizedBaseJob
from fedml.computing.scheduler.comm_utils import sys_utils
from typing import List, Dict
from os.path import expanduser


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

    def run(self):
        job_yaml_obj = self.load_yaml_config(self.job_yaml_absolute_path)
        job_yaml_obj[TrainJob.TRAIN_JOB_INPUTS_CONFIG] = json.dumps(self.input_data_list)
        job_yaml_obj[TrainJob.RUN_API_KEY_CONFIG] = sys_utils.random1(f"FEDML_NEXUS@{self.job_api_key}", "FEDML@88119999GREAT")
        self.generate_yaml_doc(job_yaml_obj, self.job_yaml_absolute_path)

        super().run()

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
            return json.loads(input_data)

    @staticmethod
    def set_outputs(output_list: List[Dict]):
        try:
            output_file = TrainJob._get_output_file()
            with open(output_file, "w") as f:
                f.write(json.dumps(output_list))
                f.write("\n\n")
            output_name = f"{TrainJob.TRAIN_JOB_OUTPUTS_KEY_PREFIX}_{self.run_id}"
            response = fedml.api.upload(data_path=output_file, name=output_name,
                                        api_key=TrainJob._get_job_api_key())
            print(f"upload response: code {response.code}, message {response.message}, data {response.data}")
        except Exception as e:
            pass

    def get_outputs(self):
        try:
            output_file = self._get_output_file()
            output_dir = os.path.dirname(output_file)
            output_name = f"{TrainJob.TRAIN_JOB_OUTPUTS_KEY_PREFIX}_{self.run_id}"
            output_data = None
            response = fedml.api.download(output_name, api_key=self.job_api_key, dest_path=output_dir)
            with open(os.path.join(output_dir, output_name), "r") as f:
                output_data = outf.readlines()
                output_json = json.loads(output_data)
                self.output_data_list.extend(output_json)
            print(f"down response: code {response.code}, message {response.message}, data {response.data}")
        except Exception as e:
            pass

        return self.output_data_list

    @staticmethod
    def _get_output_file():
        home_dir = expanduser("~")
        cache_dir = os.path.join(home_dir, ".cache")
        output_file = os.path.join(cache_dir, "fedml_jobs", str(uuid.uuid4()))

    @staticmethod
    def _get_job_api_key():
        return os.environ.get(TrainJob.RUN_API_KEY_ENV)



