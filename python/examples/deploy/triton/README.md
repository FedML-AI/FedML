# Prerequite

1. Inside config.yaml, change the 
`inference_image_name: "fedml/fedml-triton"`
to your docker image name

2. On your host machine, if you have a model model_repository that you want to mount to the container.
You need to change
`data_cache_dir: "/home/raphael/Triton/server/docs/examples/model_repository"`
to your own local directory

3. The http port for triton inference server is default to 8000, if you would prefer another port:
change
`port_inside_container: 8000`
to your inference port

# Create a new model cards with a configuration file
Note that $model_name need to be the same with the name of model in triton server.
e.g. 
```
+----------------------+---------+--------+
| Model                | Version | Status |
+----------------------+---------+--------+
| densenet_onnx        | 1       | READY  |
| inception_graphdef   | 1       | READY  |
| simple               | 1       | READY  |
| simple_dyna_sequence | 1       | READY  |
| simple_identity      | 1       | READY  |
| simple_int8          | 1       | READY  |
| simple_sequence      | 1       | READY  |
| simple_string        | 1       | READY  |
+----------------------+---------+--------+
```
if you want to serve `simple`, then: `model_name=simple`

```sh
cd FedML/python/fedml/serving/example/triton/
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
  fedml model deploy --name my_first_model --master_ids $master_id --worker_ids $client_id
  ```
 - Result
    
    See the deployment result in https://tensoropera.ai

- OPT2: Deploy - UI
    
    Follow the instructions on https://tensoropera.ai
