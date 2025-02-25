## Create a model card at local
First, create a model card at local
```bash
fedml model create -n scalellm -cf scalellm.yaml
```

## Low Code UI Deploy
Push the model to nexus ai platform
```bash
fedml model push -n scalellm
```
Do the following docs to deploy the model on nexus ai platform
https://docs-dev.fedml.ai/deploy/low_code_ui

## CLI Deploy
### Deploy to current machine 
Docs: https://docs-dev.fedml.ai/deploy/deploy_local
```bash
fedml model deploy -n scalellm --local
```

### Deploy to On-premise  
Docs: https://docs-dev.fedml.ai/deploy/deploy_on_premise
```bash
fedml device bind $api_key
```
```bash
fedml model deploy -n my_model -m $master_ids -w $worker_ids
```

### Deploy to GPU Cloud  
Docs: https://docs-dev.fedml.ai/deploy/deploy_cloud

Change the `scalellm.yaml` file, adding following lines
```yaml
computing:
  minimum_num_gpus: 1           # minimum # of GPUs to provision
  maximum_cost_per_hour: $3000   # max cost per hour for your job per gpu card
  #allow_cross_cloud_resources: true # true, false
  #device_type: CPU              # options: GPU, CPU, hybrid
  resource_type: A100-80G       # e.g., A100-80G,
  # please check the resource type list by "fedml show-resource-type"
  # or visiting URL: https://tensoropera.ai/accelerator_resource_type
```

```bash
fedml model deploy -n scalellm
```