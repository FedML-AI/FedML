import tensorflow_federated as tff


def download_and_save_stackoverflow():
    tff.simulation.datasets.stackoverflow.load_data(cache_dir='./')


def download_and_save_word_counts():
    tff.simulation.datasets.stackoverflow.load_word_counts(cache_dir='./')


def download_and_save_tag_counts():
    tff.simulation.datasets.stackoverflow.load_tag_counts(cache_dir='./')


if __name__ == "__main__":
    download_and_save_stackoverflow()
    download_and_save_word_counts()
    download_and_save_tag_counts()
