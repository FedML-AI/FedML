import os
import sys

from fedml_api.data_preprocessing.fednlp.base.base_raw_data_loader import BaseRawDataLoader
from fedml_api.data_preprocessing.fednlp.base.base_client_data_loader import BaseClientDataLoader


class RawDataLoader(BaseRawDataLoader):
    def __init__(self, data_path):
        super().__init__(data_path)
        self.task_type = "classification"
        self.target_vocab = None

    def data_loader(self):
        if len(self.X) == 0 or len(self.Y) == 0 or self.target_vocab is None:
            X = []
            Y = []
            for root1, dirs, _ in os.walk(self.data_path):
                for dir in dirs:
                    for root2, _, files in os.walk(os.path.join(root1, dir)):
                        for file_name in files:
                            file_path = os.path.join(root2, file_name)
                            X.extend(self.process_data(file_path))
                            Y.append(dir)
            self.X, self.Y = X, Y
            self.target_vocab = {key: i for i, key in enumerate(set(Y))}
            index_list = [i for i in range(len(self.X))]
            self.attributes = {"index_list": index_list}
        return {"X": self.X, "Y": self.Y, "target_vocab": self.target_vocab, "task_type": self.task_type,
                "attributes": self.attributes}

    #remove header
    def remove_header(self, lines):
        for i in range(len(lines)):
            if(lines[i] == '\n'):
                start = i+1
                break
        new_lines = lines[start:]
        return new_lines

    def process_data(self, file_path):
        X = []
        with open(file_path, "r", errors='ignore') as f:
            document = ""
            content = f.readlines()
            content = self.remove_header(content)

            for i in content:
                temp = i.lstrip("> ").replace("/\\","").replace("*","").replace("^","")
                document = document + temp

            X.append(document)
        return X

class ClientDataLoader(BaseClientDataLoader):

    def __init__(self, data_path, partition_path, client_idx=None, partition_method="uniform", tokenize=False):
        data_fields = ("X", "Y")
        super().__init__(data_path, partition_path, client_idx, partition_method, tokenize, data_fields)
        if self.tokenize:
            self.tokenize_data()

    def tokenize_data(self):
        tokenizer = self.spacy_tokenizer.en_tokenizer

        def __tokenize_data(data):
            for i in range(len(self.data["X"])):
                data["X"][i] = [str(token) for token in tokenizer(data["X"][i])]
                data["Y"][i] = [str(token) for token in tokenizer(data["Y"][i])]

        __tokenize_data(self.train_data)
        __tokenize_data(self.test_data)


# if __name__ == "__main__":
#     data_file_path = '../../../../data/fednlp/text_classification/20Newsgroups/20news-18828'
#     data_loader = RawDataLoader(data_file_path)
#     results = data_loader.data_loader()
#     uniform_partition_dict = uniform_partition(results["attributes"]["index_list"])
#
#     pickle.dump(results, open("20new_data_loader.pkl", "wb"))
#     pickle.dump({"uniform": uniform_partition_dict}, open("20news_partition.pkl", "wb"))
#
#     print("done")