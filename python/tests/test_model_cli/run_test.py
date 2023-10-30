import os
import uuid

import fedml
from fedml.api.modules import model


def test_model_create_push(config_version="release"):
    cur_dir = os.path.dirname(__file__)
    model_config = os.path.join(cur_dir, "llm_deploy", "serving.yaml")
    model_name = f"test_model_{str(uuid.uuid4())}"
    fedml.set_env_version(config_version)
    model.create(model_name, model_config=model_config)
    model.push(
        model_name, api_key="10e87dd6d6574311a80200455e4d9b30",
        tag_list=[{"tagId": 147, "parentId": 3, "tagName": "LLM"}])


if __name__ == "__main__":
    print("Hi everyone, I am testing the model cli.\n")

    test_model_create_push()

