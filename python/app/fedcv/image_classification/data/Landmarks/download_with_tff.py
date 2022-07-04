# in python3.6 logging.handlers should be imported here for tff

import tensorflow_federated as tff

tff.federated_computation(lambda: "Hello, World!")()

gld_23k_train, gld_23k_test = tff.simulation.datasets.gldv2.load_data(4, "cache", True)


print(
    "len of gld_23k_train: "
    + str(len(gld_23k_train))
    + ", len of gld_23k_train: "
    + str(len(gld_23k_test))
)
