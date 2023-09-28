## Option 1: Deploy locally
- Using fedml model deploy command with --local tag to deploy. Use -n to indicate the model name, and -cf to indicate the configuration file.
Use -cf to indicate the configuration file.
    ```sh
    fedml model deploy -n llm -cf llm.yaml --local
    ```
- After successfully deployed, you can test the model by sending a request to the local server.
    ```sh
    #INFO:     Uvicorn running on http://0.0.0.0:2345 (Press CTRL+C to quit)
    curl -XPOST localhost:2345/predict -d '{"text": "Hello"}'
    ```

## Option 2: Deploy to the Cloud (Using fedml®launch platform)
- Uncomment the following line in llm.yaml

    For information about the configuration, please refer to fedml ® launch.
    ```yaml
    # computing:
    #   minimum_num_gpus: 1
    #   maximum_cost_per_hour: $1
    #   resource_type: A100-80G 
    ```
- Remove the -l tag, and run the fedml model deploy command again
    ```sh
    fedml model deploy -n llm -cf llm.yaml
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

    - Option 1: Pass the worker id and master id using args
  ```sh
  fedml model deploy --name llm --master_ids $master_id --worker_ids $client_id --user_id $usr_id --api_key $api_key
  ```
    - Option 2: Pass the worker id and master id using environment variables
    ```sh
  export FEDML_USER_ID=YOUR_USER_ID
  export FEDML_API_KEY=YOUR_API_KEY
  export FEDML_MODEL_SERVE_MASTER_DEVICE_IDS=YOUR_MASTER_DEVICE_ID
  export FEDML_MODEL_SERVE_WORKER_DEVICE_IDS=YOUR_WORKER_DEVICE_IDS
    ```
  ```sh
  fedml model deploy --name llm
  ```
 - Result
    
    See the deployment result in https://open.fedml.ai
