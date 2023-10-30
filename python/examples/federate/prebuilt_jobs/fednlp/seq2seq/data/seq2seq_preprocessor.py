from fedml.data.fednlp.base.preprocess.base_example import Seq2SeqInputExample
from fedml.data.fednlp.base.preprocess.base_preprocessor import BasePreprocessor
from .seq2seq_data_utils import (
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
                    encoder_tokenizer, decoder_tokenizer, self.args, examples, mode,
                )
