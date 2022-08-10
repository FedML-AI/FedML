# Install FedML
```
pip install fedml
```

# Config MLOps parameters
You may set the following parameters in the 'fedml_config.yaml' file 
to upload metrics and logs to MLOps (open.fedml.ai)
```
enable_tracking: true
mlops_api_key: your_api_key
mlops_project_name: your_project_name
mlops_run_name: your_run_name_prefix
```

# Login to MLOps 
You may run the following command to login to MLOps (open.fedml.ai),
then simulation metrics and logs will be uploaded to MLOps.
```
fedml login userid(or API Key) -c -r edge_simulator
```

# Run the example (one line API)
```
python torch_fedavg_mnist_lr_one_line_example.py --cf fedml_config.yaml
```

# Run the example (step by step APIs)
```
python torch_fedavg_mnist_lr_step_by_step_example.py --cf fedml_config.yaml
```

# Run the example (custom dataset and model)
```
python torch_fedavg_mnist_lr_custum_data_and_model_example.py --cf fedml_config.yaml
```
