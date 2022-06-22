
# Customizing Data Loader for Your Own Dataset

Datasets are loaded by data_loader.py (located under [https://github.com/FedML-AI/FedML-refactor/tree/master/python/fedml/data](https://github.com/FedML-AI/FedML-refactor/tree/master/python/fedml/data). When your customized data loader follows the following data structure, FedML Parrot framework can process it without any source code change.

|__Params__ | |
|-|-|
| client_number | The number of clients in total. |
| train_data_num | The number of training samples in total. |
| test_data_num | The number of test samples in total. |
| train_data_global | Global train dataset in the form of pytorch [Dataloader](https://pytorch.org/docs/stable/data.html).|
| test_data_global | Global test dataset, in the form of pytorch [Dataloader](https://pytorch.org/docs/stable/data.html).|
| train_data_local_num_dict | Deprecated, will be removed later.  |
| train_data_local_dict | A dictionary to index the dataloader for each client. The key is the client index, and the value is the client's local data in the form of pytorch [Dataloader](https://pytorch.org/docs/stable/data.html). |
| test_data_local_dict | A dictionary to index the dataloader for each client. The key is the client index, and the value is the client's local data in the form of pytorch [Dataloader](https://pytorch.org/docs/stable/data.html). |
| class_num | The number of classes, normally used for determining the dimension of the output layer for classification task. |

Taking the simplest MNIST as example, the form of the return is as follows.

```python
logger.info("load_data. dataset_name = %s" % dataset_name)
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
) = fedml.data.load(
    args.batch_size,
    train_path=args.data_cache_dir + "/MNIST/train",
    test_path=args.data_cache_dir + "/MNIST/test",
)
```

For more examples, please read through [https://github.com/FedML-AI/FedML-refactor/blob/master/python/fedml/data/data_loader.py](https://github.com/FedML-AI/FedML-refactor/blob/master/python/fedml/data/data_loader.py).
