import os

import fedml

if __name__ == "__main__":
    print("Hi everyone, I am testing the API for fedml.mlops.log")

    os.environ['FEDML_CURRENT_RUN_ID'] = '1000'
    fedml.set_env_version("dev")

    fedml.mlops.log({"ACC": 0.1})
    fedml.mlops.log({"acc": 0.11})
    fedml.mlops.log({"acc": 0.2})
    fedml.mlops.log({"acc": 0.3})

    fedml.mlops.log({"acc": 0.31}, step=1)
    fedml.mlops.log({"acc": 0.32, "x_index": 2}, step=2, customized_step_key="x_index")
    fedml.mlops.log({"loss": 0.33}, customized_step_key="x_index", commit=False)
    fedml.mlops.log({"acc": 0.34}, step=4, customized_step_key="x_index", commit=True)

