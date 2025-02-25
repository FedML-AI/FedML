# Prerequisite

1. Change the bootstrap under /setup/bootstrap.sh, currently int the example
we started a fastapi service.
```bash
nohup python3 ./micro_service_a.py &
```
2. Inside main_entry.py, the `predict` method will redirect the request to micro_service_a
3. Inside main_entry.py, the `ready` method will send health check to micro_service_a

# Create a new model cards with a configuration file
```sh
cd FedML/python/fedml/serving/example/multi_service/
fedml model create --name $model_name --config_file config.yaml
```

## On-premsie Deploy
Register an account on TensorOpera website: https://tensoropera.ai

You will have a user id and api key, which can be found in the profile page.

- Devices Login
    ```sh
    fedml login $Your_UserId_or_ApiKey
    ```
    You will see your FedML Edge Master and Worker ID in the terminal,
    for example:
    ```
    Congratulations, your device is connected to the FedML MLOps platform successfully!
    Your FedML Edge ID is 32314, unique device ID is 0xxxxxxxx@MacOS.Edge.Device, 
    master deploy ID is 31240, worker deploy ID is 31239
    ```
    Here the master id is 31240, and worker ID is 31239.
    

- Push model card
    ```sh
    fedml model push --name $model_name
    ```

- OPT1: Deploy - CLI
  ```sh
  fedml model deploy --name $model_name --master_ids $master_id --worker_ids $client_id
  ```
 - Result
    
    See the deployment result in https://tensoropera.ai

- OPT2: Deploy - UI
    
    Follow the instructions on https://tensoropera.ai
