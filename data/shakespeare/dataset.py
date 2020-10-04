import tensorflow_federated as tff


def download_and_save_shakespeare():
    tff.simulation.datasets.shakespeare.load_data(cache_dir='./')


if __name__ == "__main__":
    download_and_save_shakespeare()