import logging
import os
import re
import string

import pandas as pd
import torch
from collections import defaultdict
from torch.utils.data import TensorDataset
from tqdm import tqdm

from ..data_preprocessing.base.base_example import SpanExtractionInputExample
from ..data_preprocessing.base.base_preprocessor import BasePreprocessor
from ..data_preprocessing.utils.span_extraction_utils import (
    squad_convert_examples_to_features,
)

customized_cleaner_dict = {}


class TrivialPreprocessor(BasePreprocessor):
    # Used for models such as LSTM, CNN, etc.
    def __init__(self, **kwargs):
        super(TrivialPreprocessor, self).__init__(**kwargs)
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

    def transform(
        self,
        context_X,
        question_X,
        y,
        y_answers,
        qas_ids=None,
        index_list=None,
        evaluate=False,
    ):
        if index_list is None:
            index_list = [i for i in range(len(context_X))]

        if qas_ids is None:
            qas_ids = index_list

        examples = self.transform_examples(
            context_X, question_X, y, y_answers, qas_ids, index_list
        )
        features = self.transform_features(examples, evaluate=evaluate)

        # Convert to Tensors and build dataset
        all_guid = torch.tensor([f.guid for f in features], dtype=torch.long)
        all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
        all_attention_masks = torch.tensor(
            [f.attention_mask for f in features], dtype=torch.long
        )
        all_token_type_ids = torch.tensor(
            [f.token_type_ids for f in features], dtype=torch.long
        )
        all_cls_index = torch.tensor([f.cls_index for f in features], dtype=torch.long)
        all_p_mask = torch.tensor([f.p_mask for f in features], dtype=torch.float)
        all_is_impossible = torch.tensor(
            [f.is_impossible for f in features], dtype=torch.float
        )

        if evaluate:
            all_feature_index = torch.arange(all_input_ids.size(0), dtype=torch.long)
            dataset = TensorDataset(
                all_guid,
                all_input_ids,
                all_attention_masks,
                all_token_type_ids,
                all_feature_index,
                all_cls_index,
                all_p_mask,
            )
        else:
            all_start_positions = torch.tensor(
                [f.start_position for f in features], dtype=torch.long
            )
            all_end_positions = torch.tensor(
                [f.end_position for f in features], dtype=torch.long
            )
            dataset = TensorDataset(
                all_guid,
                all_input_ids,
                all_attention_masks,
                all_token_type_ids,
                all_start_positions,
                all_end_positions,
                all_cls_index,
                all_p_mask,
                all_is_impossible,
            )
        return examples, features, dataset

    def transform_examples(
        self, context_X, question_X, y, y_answers, qas_ids, index_list
    ):
        examples = list()
        for c, q, a, a_t, qas_id, idx in tqdm(
            zip(context_X, question_X, y, y_answers, qas_ids, index_list),
            desc="trasforming examples",
        ):
            # ignore the qa pair which doesn't have answer
            if c[a[0] : a[1]].strip() == "":
                continue
            # answers = [{"text": c[a[0]:a[1]], "answer_start": a[0]}]
            answers = [{"text": a_t}]
            example = SpanExtractionInputExample(
                guid=int(idx),
                qas_id=qas_id,
                question_text=q,
                context_text=c,
                # answer_text=c[a[0]:a[1]],
                start_position_character=a[0],
                answer_text=a_t,
                title=None,
                is_impossible=False,
                answers=answers,
            )
            examples.append(example)
        return examples

    def transform_features(self, examples, evaluate=False, no_cache=False):
        """
        Converts a list of examples to a TensorDataset containing InputFeatures. Caches the InputFeatures.

        Utility function for train() and eval() methods. Not intended to be used directly.
        """

        tokenizer = self.tokenizer
        args = self.args

        if not no_cache:
            no_cache = args.no_cache

        if not no_cache:
            os.makedirs(self.args.cache_dir, exist_ok=True)

        mode = "dev" if evaluate else "train"
        cached_features_file = os.path.join(
            args.cache_dir,
            "cached_{}_{}_{}_{}".format(
                mode, args.model_type, args.max_seq_length, len(examples)
            ),
        )

        if os.path.exists(cached_features_file) and (
            (not args.reprocess_input_data and not no_cache)
            or (mode == "dev" and args.use_cached_eval_features)
        ):
            features = torch.load(cached_features_file)
            logging.info(f" Features loaded from cache at {cached_features_file}")
        else:
            logging.info(" Converting to features started.")

            features = squad_convert_examples_to_features(
                examples=examples,
                tokenizer=tokenizer,
                max_seq_length=args.max_seq_length,
                doc_stride=args.doc_stride,
                max_query_length=args.max_query_length,
                is_training=not evaluate,
                tqdm_enabled=not args.silent,
                threads=args.process_count,
                args=args,
            )
        return features
