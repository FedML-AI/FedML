# Create a new model cards with a configuration file
```sh
cd FedML/python/fedml/serving/example/streaming/
fedml model create --name my_first_model --config_file config.yaml
```

## Option 1: Deploy locally
- Using fedml model deploy command with --local tag to deploy. Use -n to indicate the model name, and -cf to indicate the configuration file.
Use -cf to indicate the configuration file.
    ```sh
    fedml model deploy -n my_first_model --local
    ```
- After successfully deployed, you can test the model by sending a request to the local server.
    ```sh
    #INFO:     Uvicorn running on http://0.0.0.0:2345 (Press CTRL+C to quit)
    curl -XPOST localhost:2345/predict -d '{"text": "Hello"}'
    ```

## Option 2: Deploy to the Cloud (Using TensorOpera®launch platform)
- Uncomment the following line in config.yaml

    For information about the configuration, please refer to fedml ® launch.
    ```yaml
    # computing:
    #   minimum_num_gpus: 1
    #   maximum_cost_per_hour: $1
    #   resource_type: A100-80G 
    ```
- Remove the -l tag, and run the fedml model deploy command again
    ```sh
    fedml model deploy -n my_first_model -cf config.yaml
    ```
## Option 3: On-premsie Deploy
Register an account on FedML website: https://open.fedml.ai

You will have a user id and api key, which can be found in the profile page.

- Devices Login
    ```sh
    fedml login $Your_UserId_or_ApiKey
    ```
    You will see your FedML Edge Worker ID in the terminal:
    ```
    Your FedML Edge ID is xxxx, unique device ID is xxx.OnPremise.Device
    ```
    Also, you can see your FedML Edge Master ID in the terminal:
    ```
    Your FedML Edge ID is xxxx, unique device ID is xxx.OnPremise.Master.Device
    ```
    
- Deploy
  ```sh
  fedml model deploy --name my_first_model --master_ids $master_id --worker_ids $client_id
  ```
 - Result
    
    See the deployment result in https://open.fedml.ai
