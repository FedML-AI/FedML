import os

from fedml_api.data_preprocessing.fednlp.base.base_raw_data_loader import BaseRawDataLoader
from fedml_api.data_preprocessing.fednlp.base.base_client_data_loader import BaseClientDataLoader


class RawDataLoader(BaseRawDataLoader):
    def __init__(self, data_path):
        super().__init__(data_path)
        self.task_type = "classification"
        self.target_vocab = None
        self.label_file_name = "sentiment_labels.txt"
        self.data_file_name = "dictionary.txt"

    def data_loader(self):
        if len(self.X) == 0 or len(self.Y) == 0 or self.target_vocab is None:
            X, Y = self.process_data(self.data_path)
            self.X, self.Y = X, Y
            index_list = [i for i in range(len(self.X))]
            self.attributes = {"index_list": index_list}
            self.target_vocab = {key: i for i, key in enumerate(set(Y))}
        return {"X": self.X, "Y": self.Y, "target_vocab": self.target_vocab, "task_type": self.task_type,
                "attributes": self.attributes}

    def label_level(self, label):
        label = float(label)
        if label >= 0.0 and label <= 0.2:
            return "very negative"
        elif label > 0.2 and label <= 0.4:
            return "negative"
        elif label > 0.4 and label <= 0.6:
            return "neutral"
        elif label > 0.6 and label <= 0.8:
            return "positive"
        else:
            return "very positive"

    def process_data(self, file_path):
        X = []
        Y = []
        label_dict = dict()
        with open(os.path.join(file_path, self.label_file_name)) as f:
            for label_line in f:
                label = label_line.split('|')
                label_dict[label[0].strip()] = label[1]

        with open(os.path.join(file_path, self.data_file_name)) as f:
            for data_line in f:
                data = data_line.strip().split("|")
                X.append(data[0].strip())
                Y.append(self.label_level(label_dict[data[1].strip()]))
        return X, Y


class ClientDataLoader(BaseClientDataLoader):

    def __init__(self, data_path, partition_path, client_idx=None, partition_method="uniform", tokenize=False):
        data_fields = ("X", "Y")
        super().__init__(data_path, partition_path, client_idx, partition_method, tokenize, data_fields)
        if self.tokenize:
            self.tokenize_data()

    def tokenize_data(self):
        tokenizer = self.spacy_tokenizer.en_tokenizer

        def __tokenize_data(data):
            for i in range(len(data["X"])):
                data["X"][i] = [str(token) for token in tokenizer(data["X"][i])]
                data["Y"][i] = [str(token) for token in tokenizer(data["Y"][i])]

        __tokenize_data(self.train_data)
        __tokenize_data(self.test_data)


# if __name__ == "__main__":
#     data_file_path = '../../../../data//fednlp/text_classification/SST-2/stanfordSentimentTreebank/'
#
#     train_data_loader = RawDataLoader(data_file_path)
#     results = train_data_loader.data_loader()
#     uniform_partition_dict = uniform_partition(results["attributes"]["index_list"])
#     pickle.dump(results, open("sst_2_data_loader.pkl", "wb"))
#     pickle.dump({"uniform": uniform_partition_dict}, open("sst_2_partition.pkl", "wb"))
#     print("done")
