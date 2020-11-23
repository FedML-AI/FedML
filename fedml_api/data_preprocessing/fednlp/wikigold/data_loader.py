from fedml_api.data_preprocessing.fednlp.base.base_raw_data_loader import BaseRawDataLoader
from fedml_api.data_preprocessing.fednlp.base.base_client_data_loader import BaseClientDataLoader
from fedml_api.data_preprocessing.fednlp.base.utils import *


class RawDataLoader(BaseRawDataLoader):
    def __init__(self, data_path):
        super().__init__(data_path)
        self.task_type = "sequence_tagging"
        self.target_vocab = None

    def data_loader(self):
        if len(self.X) == 0 or len(self.Y) == 0 or len(self.target_vocab) == 0:
            X, Y = self.process_data(self.data_path)
            self.X, self.Y = X, Y
            index_list = [i for i in range(len(self.X))]
            self.attributes = {"index_list": index_list}
            self.target_vocab = build_vocab(Y)

        return {"X": self.X, "Y": self.Y, "target_vocab": self.target_vocab, "task_type": self.task_type,
                "attributes": self.attributes}

    def process_data(self, file_path):
        X = []
        Y = []
        single_x = []
        single_y = []
        with open(file_path, "r") as f:
            for line in f:
                line = line.strip()
                if line:
                    token, label = line.split(" ")
                    single_x.append(token)
                    single_y.append(label)
                else:
                    if len(single_x) != 0 and len(single_y) == len(single_y):
                        X.append(single_x.copy())
                        Y.append(single_y.copy())
                    single_x.clear()
                    single_y.clear()
        return X, Y


class ClientDataLoader(BaseClientDataLoader):
    def __init__(self, data_path, partition_path, client_idx=None, partition_method="uniform", tokenize=False):
        data_fields = ("X", "Y")
        super().__init__(data_path, partition_path, client_idx, partition_method, tokenize, data_fields)




# if __name__ == "__main__":
#     data_file_path = "../../../../data/fednlp/sequence_tagging/wikigold/wikigold/CONLL-format/data/wikigold.conll.txt"
#     data_loader = RawDataLoader(data_file_path)
#     results = data_loader.data_loader()
#     uniform_partition_dict = uniform_partition(results["attributes"]["index_list"])
#     pickle.dump(results, open("wikigold_data_loader.pkl", "wb"))
#     pickle.dump({"uniform": uniform_partition_dict}, open("wikigold_partition.pkl", "wb"))
#     print("done")
