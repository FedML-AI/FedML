import tensorflow_federated as tff

def download_and_save_federated_shakespeare():
    shakespeare_train, shakespeare_test = tff.simulation.datasets.shakespeare.load_data(cache_dir='./')
    
"""
#with Tensorflow dependencies, you can run this python script to process the data from Tensorflow Federated locally:
python dataset.py

Before downloading, please install TFF as its official instruction:
pip install --upgrade tensorflow_federated
"""
if __name__ == "__main__":
    download_and_save_federated_shakespeare()