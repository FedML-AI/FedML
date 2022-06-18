import logging
import os
import re
import string

import pandas as pd
import torch
from torch.utils.data import TensorDataset

from ..data_preprocessing.base.base_example import SeqTaggingInputExample
from ..data_preprocessing.base.base_preprocessor import BasePreprocessor
from ..data_preprocessing.utils.seq_tagging_utils import convert_examples_to_features

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
        all_label_ids = torch.tensor([f.label_ids for f in features], dtype=torch.long)
        # if self.args.onnx:
        #     return all_label_ids

        dataset = TensorDataset(
            all_guid, all_input_ids, all_input_mask, all_segment_ids, all_label_ids
        )
        return examples, features, dataset

    def transform_examples(self, X, y, index_list):
        data = list()
        for sent_id, words, labels in zip(index_list, X, y):
            assert len(words) == len(labels)
            for word, label in zip(words, labels):
                data.append([sent_id, word, label])

        df = pd.DataFrame(data, columns=["sentence_id", "words", "labels"])

        examples = []
        for sentence_id, sentence_df in df.groupby(["sentence_id"]):
            words = sentence_df["words"].tolist()
            if self.text_cleaner:
                words = self.text_cleaner(words)

            examples.append(
                SeqTaggingInputExample(
                    guid=sentence_id, words=words, labels=sentence_df["labels"].tolist()
                )
            )

        return examples

    def transform_features(self, examples, evaluate=False, no_cache=False):
        """
        Reads data_file and generates a TensorDataset containing InputFeatures. Caches the InputFeatures.
        Utility function for train() and eval() methods. Not intended to be used directly.

        Args:
            data: Path to a .txt file containing training or evaluation data OR a pandas DataFrame containing 3 columns - sentence_id, words, labels.
                    If a DataFrame is given, each sentence should be split into words, with each word assigned a tag, and with all words from the same sentence given the same sentence_id.
            evaluate (optional): Indicates whether the examples are for evaluation or for training.
            no_cache (optional): Force feature conversion and prevent caching. I.e. Ignore cached features even if present.

        """  # noqa: ignore flake8"

        process_count = self.args.process_count

        tokenizer = self.tokenizer
        args = self.args

        self.args.labels_list = list(self.label_vocab.keys())

        if not no_cache:
            no_cache = args.no_cache

        mode = "dev" if evaluate else "train"

        cached_features_file = os.path.join(
            args.cache_dir,
            "cached_{}_{}_{}_{}_{}".format(
                mode,
                args.model_type,
                args.max_seq_length,
                self.args.num_labels,
                len(examples),
            ),
        )
        if not no_cache:
            os.makedirs(self.args.cache_dir, exist_ok=True)

        if os.path.exists(cached_features_file) and (
            (not args.reprocess_input_data and not no_cache)
            or (mode == "dev" and args.use_cached_eval_features and not no_cache)
        ):
            features = torch.load(cached_features_file)
            logging.info(f" Features loaded from cache at {cached_features_file}")
        else:
            logging.info(" Converting to features started.")
            features = convert_examples_to_features(
                examples,
                self.args.labels_list,
                self.args.max_seq_length,
                self.tokenizer,
                # XLNet has a CLS token at the end
                cls_token_at_end=bool(args.model_type in ["xlnet"]),
                cls_token=tokenizer.cls_token,
                cls_token_segment_id=2 if args.model_type in ["xlnet"] else 0,
                sep_token=tokenizer.sep_token,
                # RoBERTa uses an extra separator b/w pairs of sentences,
                # cf. github.com/pytorch/fairseq/commit/1684e166e3da03f5b600dbb7855cb98ddfcd0805
                sep_token_extra=bool(args.model_type in ["roberta"]),
                # PAD on the left for XLNet
                pad_on_left=bool(args.model_type in ["xlnet"]),
                pad_token=tokenizer.convert_tokens_to_ids([tokenizer.pad_token])[0],
                pad_token_segment_id=4 if args.model_type in ["xlnet"] else 0,
                pad_token_label_id=self.args.pad_token_label_id,
                process_count=process_count,
                silent=args.silent,
                use_multiprocessing=args.use_multiprocessing,
                chunksize=args.multiprocessing_chunksize,
            )

            if not no_cache:
                torch.save(features, cached_features_file)

        return features
