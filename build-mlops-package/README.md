## Outline
1. Update the Configuration Files
2. Build Server and Client Package for the FedML MLOps Platform
3. Explanation of the Configurations
4. User Guide for the Workflow in FedML MLOps Platform (open.fedml.ai)

## 1. Update the Configuration Files
```
cd fedml/build-mlops-package
```

Before building the packages for the MLOps Platform, please customize the configurations for your client and server in the following files:
   
   mlops-core/fedml-client/package/conf/fedml.yaml
   
   mlops-core/fedml-server/package/conf/fedml.yaml

Note:

- See Section 3 for the explanation of the configuration.
   
- For most of the algorithms, these two files should be the same. So you can just edit one of them and copy the editable one to the other)


## 2. Build Server and Client Package for the FedML MLOps Platform
```
cd fedml/build-mlops-package
./build.sh
```

## 3. Explanation of the Configurations

The configuration file includes four sections: `model_config`, `hyperparameters_config`, `entry_config`, `entry_arguments`.

- `model_config`: set your model name such as resnet56, LR, etc. Each value will be replaced with a real-time parameter edited from the MLOps platform (see the figure below).

![MLOps Configuration](images/mlops_pkg_config.png)

- `hyperparameters_config`: section, set your hyper-parameters such as learning rate, epochs, etc. 
  These values will also be replaced with real-time parameters edited from the MLOps platform.
  In addition, our MLOps has some "internal variables" received from the UI (see the UI below). Currently, we support the following internal variables:
   
   - ${FEDSYS.CLIENT_NUM}: an integer representing the number of user-selected devices in the MLOps UI.
   - ${FEDSYS_CLIENT_ID_LIST}: client list in one entire Federated Learning flow
   - ${FEDSYS_RUN_ID}: a run id represented one entire Federated Learning flow
   - ${FEDSYS_PRIVATE_LOCAL_DATA}: private local data path in the Federated Learning client
   - ${FEDSYS_SYNTHETIC_DATA_URL}: synthetic data URL from the server, if this value is not null, the client will download data from this URL to use as
   federated training dataset
   - ${FEDSYS_IS_USING_LOCAL_DATA}: whether use private local data as the federated training dataset 

![MLOps Configuration](images/mlops_pkg_start_run.png)


- `entry_config`: set your entry file of the source code in FedML Open Source Library, e.g., `fedml/fedml_experiments/distributed/fedavg_cross_silo/main_fedavg_cross_silo.py`


- `entry_arguments`: the value of each argument is represented with a key from model_config, hyperparameters_config, or internal variables. All the argument variable values
   will be replaced with their actual value. e.g., `model: ${model_config.modelName}` means that the value
   corresponding to the argument name 'model' will be replaced with the value corresponding to the key 'modelName' from the 'model_config' section.


## 4. User Guide for the Workflow in FedML MLOps Platform (open.fedml.ai):
After you complete your configuration and build the packages, next you may start a Federated Learning flow by the following steps:
1. Create your configuration for Federated learning on the Configurations page
2. On the above configuration page, you may upload the server package by choosing the following file: dist-package/server/package.zip.
   And also, you may upload the client package by choosing the following file: dist-package/client/package.zip.
3. Fill in the configuration name
4. Save your configuration
5. Creat run in your project
6. Choose your configuration on the create run page
7. Start your run.
   In our platform, your running job will be scheduled to cluster to do the following tasks:
   1) build a server docker image and push it to the Docker registry server.
   2) build a client docker image and push it to the Docker registry server.
   3) run server docker instance as Federated Learning aggregator.
   4) send the MQTT message to the client agent which is started by your above run.sh command
   5) when the client agent received the above MQTT message.
      It will open one client docker instance as Federated Learning client or edge device
   6) At this point, Federated Learning flow will be automatically scheduled among clients and aggregator 
      (the algorithmic workflow depends on the FL algorithm you customized; by default, it is FedAvg)
   
8. Wait for a while. You may review the run overview, device status,
   training result, system performance, and models, including client models and aggregated models
   
For more detailed guidance, please refer to [https://doc.fedml.ai/user_guide/mlops/mlops_workflow_step_by_step.html](https://doc.fedml.ai/user_guide/mlops/mlops_workflow_step_by_step.html).
