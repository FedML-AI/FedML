
# Make your own workflow with multiple jobs
## Define the job yaml
```
    working_directory = os.path.dirname(os.path.abspath(__file__))
    deploy_image_job_yaml = os.path.join(working_directory, "deploy_image_job.yaml")
    deploy_3d_job_yaml = os.path.join(working_directory, "deploy_3d_job.yaml")
    train_job_yaml = os.path.join(working_directory, "train_job.yaml")
```

## If needed, we may load the job yaml and change some config items.
``` 
    deploy_image_job_yaml_obj = DeployImageJob.load_yaml_config(deploy_image_job_yaml)
    deploy_3d_job_yaml_obj = DeployImageJob.load_yaml_config(deploy_3d_job_yaml)
    train_job_yaml_obj = DeployImageJob.load_yaml_config(train_job_yaml)
    # deploy_image_job_yaml_obj["computing"]["resource_type"] = "A100-80GB-SXM"
    # deploy_image_job_yaml_obj["computing"]["device_type"] = "GPU"
    # DeployImageJob.generate_yaml_doc(deploy_image_job_yaml_obj, deploy_image_job_yaml)
```

## Generate the job object
```
    deploy_image_job = DeployImageJob(name="deploy_image_job", job_yaml_absolute_path=deploy_image_job_yaml)
    deploy_3d_job = Deploy3DJob(name="deploy_3d_job", job_yaml_absolute_path=deploy_3d_job_yaml)
    train_job = TrainJob(name="train_job", job_yaml_absolute_path=train_job_yaml)
```

## Define the workflow
```
    workflow = Workflow(name="workflow_with_multi_jobs", loop=False)
```

## Add the job object to workflow and set the dependency (DAG based).
```    
   workflow.add_job(deploy_image_job)
    #workflow.add_job(deploy_3d_job, dependencies=[deploy_image_job])
    workflow.add_job(train_job, dependencies=[deploy_image_job])
```

## Run workflow
```
    workflow.run()
```

## After the workflow finished, print the graph, nodes and topological order
```
    print("graph", workflow.metadata.graph)
    print("nodes", workflow.metadata.nodes)
    print("topological_order", workflow.metadata.topological_order)
    print("loop", workflow.loop)
```