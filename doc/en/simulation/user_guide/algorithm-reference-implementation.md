# Algorithm Reference Implementations

FedML Parrot supports representative algorithms in different communication topologies (as the figure shown below), including Fedvg, FedOpt (ICLR 2021), FedNova (NeurIPS 2020), FedGKT (NeurIPS 2020), Decentralized FL, Vertical FL, Hierarchical FL, FedNAS, and Split Learning.

<img src="./../../_static/image/fedml-topology.png" alt="drawing" style="width:800px;"/>

The API for each algorithm is as simple as follows:

```Python
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

    # start training (use "SimulatorSingleProcess" for single process-based simulation)
    simulator = SimulatorMPI(args, device, dataset, model)
    simulator.run()
```

To specify the algorithm, please change `federated_optimizer` at `fedml_config.yaml`. For example, the following `fedml_config.yaml` will enable FedOpt algorithm.

```yaml
train_args:
  federated_optimizer: "FedOpt"
  client_id_list: "[]"
  client_num_in_total: 1000
  client_num_per_round: 10
  comm_round: 200
  epochs: 1
  batch_size: 10
  client_optimizer: sgd
  learning_rate: 0.03
  weight_decay: 0.001
```

For more details, please heck [FedML Parrot examples](./user_guide/examples.md).

