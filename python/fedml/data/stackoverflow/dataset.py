import tensorflow_federated as tff


def download_and_save_stackoverflow():
    tff.simulation.datasets.stackoverflow.load_data(cache_dir="./")


def download_and_save_word_counts():
    tff.simulation.datasets.stackoverflow.load_word_counts(cache_dir="./")


def download_and_save_tag_counts():
    tff.simulation.datasets.stackoverflow.load_tag_counts(cache_dir="./")


"""
#with Tensorflow dependencies, you can run this python script to process the data from Tensorflow Federated locally:
python dataset.py

Before downloading, please install TFF as its official instruction:
pip install --upgrade tensorflow_federated
"""
if __name__ == "__main__":
    download_and_save_stackoverflow()
    download_and_save_word_counts()
    download_and_save_tag_counts()
