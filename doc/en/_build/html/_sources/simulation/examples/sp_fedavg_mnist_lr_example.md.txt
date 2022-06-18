# Simulation with a Single Process (Standalone)

Simulating FL using a single process in your personal laptop or server. This is helpful when researchers hope to try a quick algorithmic idea in small synthetic datasets (MNIST, Shakespeare, etc.) and small models (ResNet-18, Logistic Regression, etc.). 

In this example, we will present how to apply the standalone simulation of FedML in the MINIST image classification **using a single process**. The complete code is available at [https://github.com/FedML-AI/FedML/tree/master/python/examples/simulation/sp_fedavg_mnist_lr_example](https://github.com/FedML-AI/FedML/tree/master/python/examples/simulation/sp_fedavg_mnist_lr_example).

## One line API

In this section, we present how to implement a stand-alone simulated version of the FedAvg algorithm running on  MNIST dataset using FedML with a single line of code.

### **Step 1. preparation**

First we should  create a new python environment with conda

```shell
conda create -n fedml python
conda activate fedml
```

Then we should make sure that we have completed the installation of fedml and the preparation of the dataset. 

```shell
python -m pip install --upgrade pip
python -m pip install fedml
```

The dataset is provided in this case, if you need to use other datasets you can refer to the detailed steps in **Get Start**.

### Step 2. setup Parameters

Once we have installed the dependent packages, we can set the parameters by editing the `fedml_config.yaml` as shown below, which defines the parameters `common_args`, `data_args`, `model_args`, `train_args`, `validation_args`, `device_args`, `tracking_args`  and `tracking_args`.

```YAML
common_args:
  training_type: "simulation"
  random_seed: 0

data_args:
  dataset: "mnist"
  data_cache_dir: "../../../data/mnist"
  partition_method: "hetero"
  partition_alpha: 0.5

model_args:
  model: "lr"

train_args:
  federated_optimizer: "FedAvg"
  client_id_list: "[]"
  client_num_in_total: 1000
  client_num_per_round: 10
  comm_round: 200
  epochs: 1
  batch_size: 10
  client_optimizer: sgd
  learning_rate: 0.03
  weight_decay: 0.001

validation_args:
  frequency_of_the_test: 5

device_args:
  using_gpu: false
  gpu_id: 0

comm_args:
  backend: "sp"

tracking_args:
  log_file_dir: ./log
  enable_wandb: false
  wandb_key: ee0b5f53d949c84cee7decbe7a629e63fb2f8408
  wandb_entity: fedml-ai
  wandb_project: simulation
  run_name: fedml_torch_fedavg_mnist_lr
```

In the current example,`common_args.training_type` is **"simulation"**, which means that the training here is a stand-alone simulation version,`data_args.dataset` is **"mnist"** corresponding to the current dataset used, `model_args.model` is **"lr"**, which means that the current model used is **LogisticRegression (lr)** . Please refer to XXX for other parameters and more detailed setup instructions.

### Step 3. training

Now that we have configured all the dependent environments, we can quickly implement the training of a federation learning model on the MNIST dataset for a single machine simulation with the following line of code:

```Nginx
python torch_fedavg_mnist_lr_one_line_example.py --cf fedml_config.yaml
```

The `--cf fedml_config.yaml` specifies the corresponding parameter settings file.

We can see the following output when the program is just running:

```Prolog
[2022-04-21 15:42:34,124] [INFO] [device.py:14:get_device] device = cpu
[2022-04-21 15:42:34,124] [INFO] [data_loader.py:19:download_mnist] ./MNIST.zip
100% [......................................................................] 128363870 / 128363870[2022-04-21 15:44:10,003] [INFO] [data_loader.py:63:load_for_simulation] load_data. dataset_name = mnist
[2022-04-21 15:45:25,817] [INFO] [model_hub.py:14:create] create_model. model_name = lr, output_dim = 10
[2022-04-21 15:45:25,818] [INFO] [model_hub.py:17:create] LogisticRegression + MNIST
[2022-04-21 15:45:25,860] [INFO] [fedavg_api.py:41:__init__] model = LogisticRegression(
  (linear): Linear(in_features=784, out_features=10, bias=True)
)
[2022-04-21 15:45:25,861] [INFO] [fedavg_api.py:50:__init__] self.model_trainer = <fedml.simulation.fedavg.my_model_trainer_classification.MyModelTrainer object at 0x000001CF758D8F88>
[2022-04-21 15:45:25,867] [INFO] [fedavg_api.py:81:train] self.model_trainer = <fedml.simulation.fedavg.my_model_trainer_classification.MyModelTrainer object at 0x000001CF758D8F88>
[2022-04-21 15:45:26,046] [INFO] [my_model_trainer_classification.py:52:train] Update Epoch: 0 [10/20 (50%)]   Loss: 2.369121
[2022-04-21 15:45:26,051] [INFO] [my_model_trainer_classification.py:52:train] Update Epoch: 0 [20/20 (100%)]  Loss: 2.222175
[2022-04-21 15:45:26,053] [INFO] [my_model_trainer_classification.py:57:train] Client Index = 0    Epoch: 0   Loss: 2.295648
[2022-04-21 15:45:26,057] [INFO] [my_model_trainer_classification.py:52:train] Update Epoch: 0 [10/80 (12%)]   Loss: 2.237719
[2022-04-21 15:45:26,059] [INFO] [my_model_trainer_classification.py:52:train] Update Epoch: 0 [20/80 (25%)]   Loss: 2.212439
[2022-04-21 15:45:26,060] [INFO] [my_model_trainer_classification.py:52:train] Update Epoch: 0 [30/80 (38%)]   Loss: 2.156245
[2022-04-21 15:45:26,062] [INFO] [my_model_trainer_classification.py:52:train] Update Epoch: 0 [40/80 (50%)]   Loss: 2.132013
[2022-04-21 15:45:26,064] [INFO] [my_model_trainer_classification.py:52:train] Update Epoch: 0 [50/80 (62%)]   Loss: 1.966263
[2022-04-21 15:45:26,067] [INFO] [my_model_trainer_classification.py:52:train] Update Epoch: 0 [60/80 (75%)]   Loss: 2.014680
[2022-04-21 15:45:26,068] [INFO] [my_model_trainer_classification.py:52:train] Update Epoch: 0 [70/80 (88%)]   Loss: 1.959164
[2022-04-21 15:45:26,070] [INFO] [my_model_trainer_classification.py:52:train] Update Epoch: 0 [80/80 (100%)]  Loss: 2.071570
[2022-04-21 15:45:26,070] [INFO] [my_model_trainer_classification.py:57:train] Client Index = 0    Epoch: 0   Loss: 2.093762
```

At the end of the program run, we can see the following output,which indicate that we have successfully finished the entire process:

```Apache
INFO:root:################Communication round : 197
INFO:root:client_indexes = [954 887 866  96 130 891 444 925 516 308]
INFO:root:client_indexes = [954 887 866  96 130 891 444 925 516 308]
INFO:root:################Communication round : 198
INFO:root:client_indexes = [276 317 986 132  19 794 485 973  28 750]
INFO:root:client_indexes = [276 317 986 132  19 794 485 973  28 750]
INFO:root:################Communication round : 199
INFO:root:client_indexes = [221 905 942 888 711 479 506 685 436 661]
INFO:root:client_indexes = [221 905 942 888 711 479 506 685 436 661]
INFO:root:################local_test_on_all_clients : 199
INFO:root:{'training_acc': 0.8190029839128179, 'training_loss': 1.7344120322737784}
INFO:root:{'test_acc': 0.8188848188848189, 'test_loss': 1.73456031979222}
```

The code for `torch_fedavg_mnist_lr_one_line_example.py` is shown below:

```Python
import fedml

if __name__ == "__main__":
    fedml.run_simulation()
```

As you can see, the `main()` function is a single line of code that implements the training of a standalone simulation of federal learning. 

### step 4. view the results

You can view the output log files in the `/log` directory under the current directory

## Step by step API

In this section, we will present how to use FedML to implement a stand-alone simulated version of the FedAvg algorithm on the MNIST dataset by calling specific functions **step by step** in the code.

First we should also complete the step1 and step2 operations in the **one line example**, and then quickly implement the federation learning model training on the MNIST dataset with the following line of code for a single machine simulation.

```Nginx
python torch_fedavg_mnist_lr_step_by_step_example.py --cf fedml_config.yaml
```

The following steps are implemented in 

```
torch_fedavg_mnist_lr_step_by_step_example.py
```

- **Init FedML framework :** initialize FedML framework, get the parameter settings of each part(`args`)

- **Init device:** set the `device` to run according to the parameters related to device settings.

- **Load data:** load the dataset information(`dataset`) and the dimension of the task output (`output_dim`)according to the parameters related to data. For example, the current dataset is a 10 classification task, so the dimension of the task output is 10.

- **Load model:** load the initialized `model` according to the parameters related to the model.

- **Start training:** initialize the standalone simulator object `Simulator(args, device, dataset, model)`  with `args`, `device`, `dataset`, `model` values, and then call run() on the object to start training.

## Custom data and model

In this section we will present how to **customize the dataset and model** using FedML based on the `Step by step example` and implement a stand-alone simulated version of the FedAvg algorithm.

First we still need to complete the first two cases step1 and step2,  and then we can quickly implement the federation learning model training on the MNIST dataset with the following line of code for stand-alone simulation:

```Nginx
python torch_fedavg_mnist_lr_custum_data_and_model_example.py --cf fedml_config.yaml
```

The `torch_fedavg_mnist_lr_custum_data_and_model_example.py` defines the 

`load_data(args)` function for loading the dataset and related information, and the 

`LogisticRegression(torch.nn. Module)` class defines the LogisticRegression model. The specific code is as follows:

```Python
def load_data(args):
    download_mnist(args.data_cache_dir)
    fedml.logger.info("load_data. dataset_name = %s" % args.dataset)

    """
    Please read through the data loader at to see how to customize the dataset for FedML framework.
    """
    (
        client_num,
        train_data_num,
        test_data_num,
        train_data_global,
        test_data_global,
        train_data_local_num_dict,
        train_data_local_dict,
        test_data_local_dict,
        class_num,
    ) = load_partition_data_mnist(
        args.batch_size,
        train_path=args.data_cache_dir + "MNIST/train",
        test_path=args.data_cache_dir + "MNIST/test",
    )
    """
    For shallow NN or linear models, 
    we uniformly sample a fraction of clients each round (as the original FedAvg paper)
    """
    args.client_num_in_total = client_num
    dataset = [
        train_data_num,
        test_data_num,
        train_data_global,
        test_data_global,
        train_data_local_num_dict,
        train_data_local_dict,
        test_data_local_dict,
        class_num,
    ]
    return dataset, class_num

class LogisticRegression(torch.nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LogisticRegression, self).__init__()
        self.linear = torch.nn.Linear(input_dim, output_dim)

    def forward(self, x):
        outputs = torch.sigmoid(self.linear(x))
        return outputs
```

`torch_fedavg_mnist_lr_custum_data_and_model_example.py` is similar to `torch_fedavg_mnist_lr_step_by_step_example.py`, the code includes the same parts,  though `torch_fedavg_mnist_lr_custum_data_and_model_example.py` loads the dataset and the model definition part using custom function and class. The code for the training process is shown below:

```Python
if __name__ == "__main__":
    # init FedML framework
    args = fedml.init()

    # init device
    device = fedml.device.get_device(args)

    # load data
    dataset, output_dim = load_data(args)

    # load model (the size of MNIST image is 28 x 28)
    model = LogisticRegression(28 * 28, output_dim)

    # start training
    simulator = Simulator(args, device, dataset, model)
    simulator.run()
```
