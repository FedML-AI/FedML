# TODO: will finish this part ASAP
import logging
import os
import re
import string

import pandas as pd
import torch
from torch.utils.data import TensorDataset

from ..data_preprocessing.base.base_example import Seq2SeqInputExample
from ..data_preprocessing.base.base_preprocessor import BasePreprocessor
from ..data_preprocessing.utils.seq2seq_utils import (
    Seq2SeqDataset,
    SimpleSummarizationDataset,
)

customized_cleaner_dict = {}


class TrivialPreprocessor(BasePreprocessor):
    # Used for models such as LSTM, CNN, etc.
    def __init__(self, **kwargs):
        super(TrivialPreprocessor, self).__init__(**kwargs)
        self.text_cleaner = customized_cleaner_dict.get(self.args.dataset, None)

    def transform(self, X, y):
        pass


class TLMPreprocessor(BasePreprocessor):
    # Used for Transformer language models (TLMs) such as BERT, RoBERTa, etc.
    def __init__(self, **kwargs):
        super(TLMPreprocessor, self).__init__(**kwargs)
        self.text_cleaner = customized_cleaner_dict.get(self.args.dataset, None)

    def transform(self, X, y, index_list=None, evaluate=False):
        if index_list is None:
            index_list = [i for i in range(len(X))]

        examples = self.transform_examples(X, y, index_list)
        features = self.transform_features(examples, evaluate)

        # for seq2seq task, transform_features func transform examples to dataset directly
        dataset = features

        return examples, features, dataset

    def transform_examples(self, X, y, index_list):
        examples = list()
        for src_text, tgt_text, idx in zip(X, y, index_list):
            examples.append(Seq2SeqInputExample(idx, src_text, tgt_text))
        return examples

    def transform_features(self, examples, evaluate=False, no_cache=False):
        encoder_tokenizer = self.tokenizer
        decoder_tokenizer = self.tokenizer
        args = self.args

        if not no_cache:
            no_cache = args.no_cache

        if not no_cache:
            os.makedirs(self.args.cache_dir, exist_ok=True)

        mode = "dev" if evaluate else "train"

        if args.dataset_class:
            CustomDataset = args.dataset_class
            return CustomDataset(
                encoder_tokenizer, decoder_tokenizer, args, examples, mode
            )
        else:
            if args.model_type in ["bart", "mbart", "marian"]:
                return SimpleSummarizationDataset(
                    encoder_tokenizer, self.args, examples, mode
                )
            else:
                return Seq2SeqDataset(
                    encoder_tokenizer,
                    decoder_tokenizer,
                    self.args,
                    examples,
                    mode,
                )
