import logging
import re
import string

import pandas as pd
import torch
from torch.utils.data import TensorDataset

from fedml.data.fednlp.base.preprocess.base_example import (
    TextClassificationInputExample,
)
from fedml.data.fednlp.base.preprocess.base_preprocessor import BasePreprocessor
from .text_classification_data_utils import (
    convert_examples_to_features,
)

customized_cleaner_dict = {}


class TrivialPreprocessor(BasePreprocessor):
    # Used for models such as LSTM, CNN, etc.
    def __init__(self, **kwargs):
        super(TrivialPreprocessor, self).__init__(kwargs)
        self.text_cleaner = customized_cleaner_dict.get(self.args.dataset, None)

    def transform(self, X, y):
        transformed_X = list()
        transformed_y = list()
        for i, single_x in enumerate(X):
            if self.text_cleaner:
                single_x = self.text_cleaner(single_x)
            x_tokens = [
                token.text.strip().lower()
                for token in self.tokenizer(single_x.strip())
                if token.text.strip()
            ]
            x_token_ids = [
                self.word_vocab[token]
                if token in self.word_vocab
                else self.word_vocab["<UNK>"]
                for token in x_tokens
            ]
            transformed_X.append(x_token_ids)
            transformed_y.append(self.label_vocab[y[i]])
        return transformed_X, transformed_y


class TLMPreprocessor(BasePreprocessor):
    # Used for Transformer language models (TLMs) such as BERT, RoBERTa, etc.
    def __init__(self, **kwargs):
        super(TLMPreprocessor, self).__init__(**kwargs)
        self.text_cleaner = customized_cleaner_dict.get(self.args.dataset, None)

    def transform(self, X, y, index_list=None, evaluate=False):
        # index_list is creat for setting guid
        if index_list is None:
            index_list = [i for i in range(len(X))]
        examples = self.transform_examples(X, y, index_list)
        features = self.transform_features(examples, evaluate=evaluate)

        all_guid = torch.tensor([f.guid for f in features], dtype=torch.long)
        all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
        all_input_mask = torch.tensor(
            [f.input_mask for f in features], dtype=torch.long
        )
        all_segment_ids = torch.tensor(
            [f.segment_ids for f in features], dtype=torch.long
        )
        all_label_ids = torch.tensor([f.label_id for f in features], dtype=torch.long)

        dataset = TensorDataset(
            all_guid, all_input_ids, all_input_mask, all_segment_ids, all_label_ids
        )

        return examples, features, dataset

    def transform_examples(self, X, y, index_list):
        # index_list is creat for setting guid
        data = [(X[i], self.label_vocab[y[i]], idx) for i, idx in enumerate(index_list)]

        df = pd.DataFrame(data)

        examples = []
        for i, (text, label, guid) in enumerate(
            zip(df.iloc[:, 0], df.iloc[:, 1], df.iloc[:, 2])
        ):
            if self.text_cleaner:
                text = self.text_cleaner(text)
            examples.append(TextClassificationInputExample(guid, text, None, label))

        return examples

    def transform_features(
        self, examples, evaluate=False, no_cache=False, silent=False
    ):
        """
        Converts a list of InputExample objects to a TensorDataset containing InputFeatures. Caches the InputFeatures.
        """
        process_count = self.args.process_count

        tokenizer = self.tokenizer
        args = self.args

        if not no_cache:
            no_cache = args.no_cache

        output_mode = "classification"

        # if not no_cache:
        #     os.makedirs(args.cache_dir, exist_ok=True)

        # mode = "dev" if evaluate else "train"
        # cached_features_file = os.path.join(
        #     args.cache_dir,
        #     "cached_{}_{}_{}_{}_{}".format(
        #         mode, args.model_type, args.max_seq_length, len(self.label_vocab), len(examples),
        #     ),
        # )
        # logging.debug("cached_features_file = %s" % str(cached_features_file))
        # logging.debug("args.reprocess_input_data = %s" % str(args.reprocess_input_data))
        # logging.debug("no_cache = %s" % str(no_cache))
        # if os.path.exists(cached_features_file) and (
        #         (not args.reprocess_input_data and not no_cache)
        #         or (mode == "dev" and args.use_cached_eval_features and not no_cache)
        # ):
        #     logging.info(cached_features_file)
        #     features = torch.load(cached_features_file)
        #     logging.debug(f" Features loaded from cache at {cached_features_file}")
        # else:
        logging.debug(" Converting to features started. Cache is not used.")

        # If labels_map is defined, then labels need to be replaced with ints
        if args.labels_map and not args.regression:
            for example in examples:
                example.label = args.labels_map[example.label]

        features = convert_examples_to_features(
            examples,
            args.max_seq_length,
            tokenizer,
            output_mode,
            # XLNet has a CLS token at the end
            cls_token_at_end=bool(args.model_type in ["xlnet"]),
            cls_token=tokenizer.cls_token,
            cls_token_segment_id=2 if args.model_type in ["xlnet"] else 0,
            sep_token=tokenizer.sep_token,
            # RoBERTa uses an extra separator b/w pairs of sentences,
            # cf. github.com/pytorch/fairseq/commit/1684e166e3da03f5b600dbb7855cb98ddfcd0805
            sep_token_extra=bool(
                args.model_type in ["roberta", "camembert", "xlmroberta", "longformer"]
            ),
            # PAD on the left for XLNet
            pad_on_left=bool(args.model_type in ["xlnet"]),
            pad_token=tokenizer.convert_tokens_to_ids([tokenizer.pad_token])[0],
            pad_token_segment_id=4 if args.model_type in ["xlnet"] else 0,
            process_count=process_count,
            multi_label=False,
            silent=args.silent or silent,
            use_multiprocessing=args.use_multiprocessing,
            sliding_window=args.sliding_window,
            flatten=not evaluate,
            stride=args.stride,
            add_prefix_space=bool(
                args.model_type in ["roberta", "camembert", "xlmroberta", "longformer"]
            ),
            # avoid padding in case of single example/online inferencing to decrease execution time
            pad_to_max_length=bool(len(examples) > 1),
            args=args,
        )
        logging.info(f" {len(features)} features created from {len(examples)} samples.")

        # if not no_cache:
        #     torch.save(features, cached_features_file)
        return features


def cleaner_sentiment140(text):
    # return text  # TODO: if you would like to skip this.
    text = re.sub(r"\&\w*;", "", text)
    text = re.sub("@[^\s]+", "", text)
    text = re.sub(r"\$\w*", "", text)
    text = text.lower()
    text = re.sub(r"https?:\/\/.*\/\w*", "", text)
    text = re.sub(r"#\w*", "", text)
    text = re.sub(r"[" + string.punctuation.replace("@", "") + "]+", " ", text)
    text = re.sub(r"\b\w{1,2}\b", "", text)
    text = re.sub(r"\s\s+", " ", text)
    text = [char for char in list(text) if char not in string.punctuation]
    text = "".join(text)
    text = text.lstrip(" ")
    return text


def cleaner_news20(text):
    text = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", text)
    text = re.sub(r"\'s", " 's", text)
    text = re.sub(r"\'ve", " 've", text)
    text = re.sub(r"n\'t", " n't", text)
    text = re.sub(r"\'re", " 're", text)
    text = re.sub(r"\'d", " 'd", text)
    text = re.sub(r"\'ll", " 'll", text)
    text = re.sub(r",", " , ", text)
    text = re.sub(r"!", " ! ", text)
    text = re.sub(r"\(", " \( ", text)
    text = re.sub(r"\)", " \) ", text)
    text = re.sub(r"\?", " \? ", text)
    text = re.sub(r"\s{2,}", " ", text)
    return text.strip().lower()


# Mapping the dataset to their specific cleaner
customized_cleaner_dict["sentiment140"] = cleaner_sentiment140
customized_cleaner_dict["news_20"] = cleaner_news20
