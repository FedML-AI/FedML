# Simulation with Message Passing Interface (MPI)

MPI-based Federated Learning for cross-GPU/CPU servers.

In this example, we will present how to apply the standalone simulation of FedML in the MINIST image classification using an **MPI-based FL simulator**. The complete code is available at [https://github.com/FedML-AI/FedML/tree/master/python/examples/simulation/mpi_torch_fedavg_mnist_lr_example](https://github.com/FedML-AI/FedML/tree/master/python/examples/simulation/mpi_torch_fedavg_mnist_lr_example).

## One line API

### Step 1. preparation

In this example, we have to complete the operation of step1 in the One line API first, and then proceed to the subsequent operations.

We use the `mpi4py` package for the MPI implementation in Python. Here you need to install `mpi4py` with `conda install` instead of `pip install` because pip install will report an error.

```shell
conda install mpi4py
```

### Step 2. setup Parameters

`config/fedml_config.yaml` is almost the same as `fedml_config.yaml` in **<u>One line API</u>** **step2**, only the backend setting in `comm_args` is different. Here comm_args.backend is "MPI", which means the program is running based on an MPI-based FL Simulator.

```yaml
comm_args:
  backend: "MPI"
  is_mobile: 0
```

### Step 3. training

Now that we have configured all the dependencies, we can quickly implement the training of the federated learning model based on MPI-based FL Simulator on the MNIST dataset with the following line of code.

```shell
# 4 means 4 processes
bash run_one_line_example.sh 4
```

Note that if you download the code on Windows and upload it to a Linux environment, you will see the following output when the program just starts running.

> run_one_line_example.sh: line 2: $' \r ':  command not found
>
> run_one_line_example.sh: line 4: $'\r': command not found
>
> expr: non-integer argument
>
> run_one_line_example.sh: line 7: $'\r': command not found
>
> run_one_line_example.sh: line 9: $'\r': command not found

The main way to deal with this is to end the program and then run the following command, `filename` here is `run_one_line_example.sh`

```shell
sed -i 's/\r$//' filename
```

When it officially starts running, you can see the real-time output of the program running on the terminal. When the process is finished and you see an output similar to the following, it means that the whole process is running successfully.

```shell
FedML-Server(0) @device-id-0 - Tue, 26 Apr 2022 02:53:22 FedAvgClientManager.py[line:71] INFO #######training########### round_id = 49
[2022-04-26 02:53:22,054] [INFO] [my_model_trainer_classification.py:56:train] Update Epoch: 0 [10/30 (33%)]    Loss: 1.775815
[2022-04-26 02:53:22,055] [INFO] [my_model_trainer_classification.py:56:train] Update Epoch: 0 [20/30 (67%)]    Loss: 1.920599
[2022-04-26 02:53:22,055] [INFO] [my_model_trainer_classification.py:56:train] Update Epoch: 0 [30/30 (100%)]   Loss: 1.839799
[2022-04-26 02:53:22,055] [INFO] [my_model_trainer_classification.py:63:train] Client Index = 3 Epoch: 0        Loss: 1.845405
FedML-Server(0) @device-id-0 - Tue, 26 Apr 2022 02:53:22 client_manager.py[line:104] INFO Sending message (type 3) to server
FedML-Server(0) @device-id-0 - Tue, 26 Apr 2022 02:53:22 client_manager.py[line:118] INFO __finish client
--------------------------------------------------------------------------
MPI_ABORT was invoked on rank 4 in communicator MPI_COMM_WORLD
with errorcode 0.

NOTE: invoking MPI_ABORT causes Open MPI to kill all MPI processes.
You may or may not see output from other processes, depending on
exactly when Open MPI kills them.
--------------------------------------------------------------------------

```

Let's have a look at the `run_one_line_example.sh` 

* When we run the above command, 4 will be assigned to the parameter `WORKER_NUM`, representing 4 client processes. 

* Executing `echo $PROCESS_NUM`  will output 5 on the terminal, representing the total number of processes is 5 (including server).
* `hostname > mpi_host_file`
* `$(which mpirun) -np $PROCESS_NUM \`
  `-hostfile mpi_host_file \`
  `python torch_fedavg_mnist_lr_one_line_example.py --cf config/fedml_config.yaml` This line of code mpirun will run the program using the mpi method and specify the parameter file with `-np $PROCESS_NUM` specifies the total number of processes in the program, `--cf config/fedml_config.yaml`, `hostname > mpi_host_file` write the hostname to the mpi_host_file fileand `-hostfile mpi_host_file` specifies the host file of the mpi

```shell
#!/usr/bin/env bash

WORKER_NUM=$1

PROCESS_NUM=`expr $WORKER_NUM + 1`
echo $PROCESS_NUM

hostname > mpi_host_file

$(which mpirun) -np $PROCESS_NUM \
-hostfile mpi_host_file \
python torch_fedavg_mnist_lr_one_line_example.py --cf config/fedml_config.yaml
```

The code for `torch_fedavg_mnist_lr_one_line_example.py` is shown below:

```python
import fedml


if __name__ == "__main__":
    fedml.run_simulation(backend="MPI")
```

### Step 4. view the results

You can view the output log files in the `/log` directory under the current directory.

## Step by step API

First, we should also complete the step1 and step2 operations in the **one-line example** and quickly implement the federation learning model training on the MNIST dataset with the following line of code for the MPI-based FL Simulator.

```shell
sh run_step_by_step_example.sh 4
```

The code of `run_step_by_step_example.sh` is as follows, which is generally the same as `run_one_line_example.sh` in this section of the **One line API**

```shell
#!/usr/bin/env bash

WORKER_NUM=$1

PROCESS_NUM=`expr $WORKER_NUM + 1`
echo $PROCESS_NUM

hostname > mpi_host_file

$(which mpirun) -np $PROCESS_NUM \
-hostfile mpi_host_file \
python torch_fedavg_mnist_lr_step_by_step_example.py --cf config/fedml_config.yaml
```

The code of `torch_fedavg_mnist_lr_step_by_step_example.py`  is shown below. We can see that the code follows the steps of the **Step by step API** in **Example: Simulate FL using a single process**.

The difference is that `simulator = SimulatorMPI(args, device, dataset, model)` is used to initialize the model object, which means that MPI-based FL Simulator is used here for training

```python
import fedml
from fedml.simulation import SimulatorMPI

if __name__ == "__main__":
    # init FedML framework
    args = fedml.init()

    # init device
    device = fedml.device.get_device(args)

    # load data
    dataset, output_dim = fedml.data.load(args)

    # load model
    model = fedml.model.create(args, output_dim)

    # start training
    simulator = SimulatorMPI(args, device, dataset, model)
    simulator.run()
```

## Custom data and model

The operation of this part is similar to the **custom data and mode**l part in example  [**<u>Simulate FL using a single process</u>**](./sp_fedavg_mnist_lr_example.md) .

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
    simulator = SimulatorMPI(args, device, dataset, model)
    simulator.run()
```



## Develop new algorithms

### Base framework

`python/examples/simulation/mpi_base_framework_example`

This is a base framework used to develop new algorithm.
You can copy this directory and modify directly. The basic message flow is workable. 
What you need to do is designing the message flow and defining the payload of each message.

As a research library, our philosophy is to give flexibility to users and avoid over-designed software patterns. 

Run the example:

```shell
sh run.sh 4
```

`run.sh`

```shell
#!/usr/bin/env bash

WORKER_NUM=$1

PROCESS_NUM=`expr $WORKER_NUM + 1`
echo $PROCESS_NUM

hostname > mpi_host_file

$(which mpirun) -np $PROCESS_NUM \
-hostfile mpi_host_file \
python mpi_base_framework_example.py --cf config/fedml_config.yaml
```

`mpi_base_framework_example.py` 

```python
import fedml
from fedml import SimulatorMPI

if __name__ == "__main__":
    # init FedML framework
    args = fedml.init()

    # start training
    simulator = SimulatorMPI(args, None, None, None)
    simulator.run()
```

You can customize **device, dataset, model** in this base framework.

### Decentralized framework 

`python/examples/simulation/mpi_decentralized_fl_example`

This is a decentralized framework used to develop new algorithm.
You can copy this directory and modify directly. The basic message flow is workable. 
What you need to do is designing the message flow and defining the payload of each message.

As a research library, our philosophy is giving flexibility to users and avoid overdesigned software pattern. 

Run the example:

```shell
sh run.sh 4
```

`run.sh`

```shell
#!/usr/bin/env bash

WORKER_NUM=$1

PROCESS_NUM=`expr $WORKER_NUM + 1`
echo $PROCESS_NUM

hostname > mpi_host_file

$(which mpirun) -np $PROCESS_NUM \
-hostfile mpi_host_file \
python mpi_decentralized_fl_example.py --cf config/fedml_config.yaml
```

`mpi_decentralized_fl_example.py` 

```python
import fedml
from fedml import SimulatorMPI

if __name__ == "__main__":
    # init FedML framework
    args = fedml.init()

    # start training
    simulator = SimulatorMPI(args, None, None, None)
    simulator.run()
```

You can customize **device, dataset, model** in this base framework.

