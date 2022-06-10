import tensorflow_federated as tff

only_digit = False


def download_and_save_federated_emnist():
    emnist_train, emnist_test = tff.simulation.datasets.emnist.load_data(
        cache_dir="./", only_digits=only_digit
    )


"""
Or with Tensorflow dependencies, you can run this to process the data from Tensorflow locally:

```
python dataset.py
```
"""
if __name__ == "__main__":
    download_and_save_federated_emnist()
