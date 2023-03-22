# FedML Octopus API Reference

The philosophy of out API design is to reduce the number of APIs as much as possible while simultaneously maintaining the flexibility.

For simplicity, FedML Parrot has only one line API as the following example:

```Python
import fedml

if __name__ == "__main__":
    fedml.run_simulation()
```

To meet the customization requests, FedML Octopus also has five lines of APIs as the following example.

The FL Client APIs:
```Python
import fedml
from fedml.cross_silo import Client

if __name__ == "__main__":
    args = fedml.init()

    # init device
    device = fedml.device.get_device(args)

    # load data
    dataset, output_dim = fedml.data.load(args)

    # load model
    model = fedml.model.create(args, output_dim)

    # start training
    client = Client(args, device, dataset, model)
    client.run()

```

The FL Server APIs:
```Python
import fedml
from fedml.cross_silo import Server

if __name__ == "__main__":
    args = fedml.init()

    # init device
    device = fedml.device.get_device(args)

    # load data
    dataset, output_dim = fedml.data.load(args)

    # load model
    model = fedml.model.create(args, output_dim)

    # start training
    server = Server(args, device, dataset, model)
    server.run()
```



For newly developed features, we will try to keep the form of these APIs and only add new arguments. 

To check the details of the latest definition of each API, the best resource is always the source code itself. Please check comments of each API at:
[https://github.com/FedML-AI/FedML/blob/master/python/fedml/__init__.py](https://github.com/FedML-AI/FedML/blob/master/python/fedml/__init__.py)
