# FedML Parrot API Reference


The philosophy of our API design is to reduce the number of APIs as much as possible while simultaneously maintaining the flexibility.

For Simplicity, FedML Parrot has only one line API as the following example:

```Python
import fedml

if __name__ == "__main__":
    fedml.run_simulation()
```

To meet the customization demands, FedML Parrot also has five lines of APIs as the following example.
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

    # start training
    simulator = SimulatorMPI(args, device, dataset, model)
    simulator.run()
```



For newly developed features, we will try to preserve the form of these APIs and only add new arguments. 

To check out the details of the latest definition of each API, the best resource is always the source code itself. Please check comments of each API at:
[https://github.com/FedML-AI/FedML/blob/master/python/fedml/__init__.py](https://github.com/FedML-AI/FedML/blob/master/python/fedml/__init__.py)
