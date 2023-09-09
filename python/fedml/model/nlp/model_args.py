import json
import os
import sys
from dataclasses import asdict, dataclass, field
from multiprocessing import cpu_count

from torch.nn import CrossEntropyLoss
from torch.utils.data import Dataset


def get_default_process_count():
    """
    Get the default number of processes to use for multi-processing tasks.

    Returns:
        int: The default process count.

    Example:
        >>> process_count = get_default_process_count()
    """
    process_count = int(cpu_count() / 2) if cpu_count() > 2 else 1
    if sys.platform == "win32":
        process_count = min(process_count, 61)
    return process_count

def get_special_tokens():
    """
    Get a list of special tokens commonly used in natural language processing tasks.

    Returns:
        List[str]: A list of special tokens.

    Example:
        >>> special_tokens = get_special_tokens()
    """
    return ["<s>", "<pad>", "</s>", "<unk>", "<mask>"]


@dataclass
class ModelArgs:
    """
    Configuration class for model training and evaluation.

    Attributes:
        adam_epsilon (float): Epsilon value for Adam optimizer. Default is 1e-8.
        best_model_dir (str): Directory to save the best model checkpoints. Default is "outputs/best_model".
        cache_dir (str): Directory for caching data. Default is "cache_dir/".
        config (dict): Additional configuration settings as a dictionary. Default is an empty dictionary.
        custom_layer_parameters (list): List of custom layer parameters. Default is an empty list.
        custom_parameter_groups (list): List of custom parameter groups. Default is an empty list.
        dataloader_num_workers (int): Number of workers for data loading. Default is determined by `get_default_process_count`.
        do_lower_case (bool): Whether to convert input text to lowercase. Default is False.
        dynamic_quantize (bool): Whether to dynamically quantize the model. Default is False.
        early_stopping_consider_epochs (bool): Whether to consider epochs for early stopping. Default is False.
        early_stopping_delta (float): Minimum change in metric value to consider for early stopping. Default is 0.
        early_stopping_metric (str): Metric to monitor for early stopping. Default is "eval_loss".
        early_stopping_metric_minimize (bool): Whether to minimize the early stopping metric. Default is True.
        early_stopping_patience (int): Number of epochs with no improvement to wait before early stopping. Default is 3.
        encoding (str): Encoding for input text. Default is None.
        eval_batch_size (int): Batch size for evaluation. Default is 8.
        evaluate_during_training (bool): Whether to perform evaluation during training. Default is False.
        evaluate_during_training_silent (bool): Whether to silence evaluation logs during training. Default is True.
        evaluate_during_training_steps (int): Frequency of evaluation steps during training. Default is 2000.
        evaluate_during_training_verbose (bool): Whether to print evaluation results during training. Default is False.
        evaluate_each_epoch (bool): Whether to perform evaluation after each epoch. Default is True.
        fp16 (bool): Whether to use mixed-precision training (FP16). Default is True.
        gradient_accumulation_steps (int): Number of gradient accumulation steps. Default is 1.
        learning_rate (float): Learning rate for training. Default is 4e-5.
        local_rank (int): Local rank for distributed training. Default is -1.
        logging_steps (int): Frequency of logging training steps. Default is 50.
        manual_seed (int): Seed for random number generation. Default is None.
        max_grad_norm (float): Maximum gradient norm for clipping gradients. Default is 1.0.
        max_seq_length (int): Maximum sequence length for input data. Default is 128.
        model_name (str): Name of the model being used. Default is None.
        model_type (str): Type of the model being used. Default is None.
        ... (other attributes)

    Methods:
        update_from_dict(new_values): Update attribute values from a dictionary.
        get_args_for_saving(): Get a dictionary of attributes suitable for saving.
        save(output_dir): Save the model configuration to a JSON file in the specified output directory.
        load(input_dir): Load the model configuration from a JSON file in the specified input directory.
    """
    adam_epsilon: float = 1e-8
    best_model_dir: str = "outputs/best_model"
    cache_dir: str = "cache_dir/"
    config: dict = field(default_factory=dict)
    custom_layer_parameters: list = field(default_factory=list)
    custom_parameter_groups: list = field(default_factory=list)
    dataloader_num_workers: int = field(default_factory=get_default_process_count)
    do_lower_case: bool = False
    dynamic_quantize: bool = False
    early_stopping_consider_epochs: bool = False
    early_stopping_delta: float = 0
    early_stopping_metric: str = "eval_loss"
    early_stopping_metric_minimize: bool = True
    early_stopping_patience: int = 3
    encoding: str = None
    eval_batch_size: int = 8
    evaluate_during_training: bool = False
    evaluate_during_training_silent: bool = True
    evaluate_during_training_steps: int = 2000
    evaluate_during_training_verbose: bool = False
    evaluate_each_epoch: bool = True
    fp16: bool = True
    gradient_accumulation_steps: int = 1
    learning_rate: float = 4e-5
    local_rank: int = -1
    logging_steps: int = 50
    manual_seed: int = None
    max_grad_norm: float = 1.0
    max_seq_length: int = 128
    model_name: str = None
    model_type: str = None
    multiprocessing_chunksize: int = 500
    n_gpu: int = 1
    no_cache: bool = False
    no_save: bool = False
    not_saved_args: list = field(default_factory=list)
    epochs: int = 1
    output_dir: str = "outputs/"
    overwrite_output_dir: bool = False
    process_count: int = field(default_factory=get_default_process_count)
    quantized_model: bool = False
    reprocess_input_data: bool = True
    save_best_model: bool = True
    save_eval_checkpoints: bool = True
    save_model_every_epoch: bool = True
    save_optimizer_and_scheduler: bool = True
    save_steps: int = 2000
    silent: bool = False
    tensorboard_dir: str = None
    thread_count: int = None
    train_batch_size: int = 8
    train_custom_parameters_only: bool = False
    use_cached_eval_features: bool = False
    use_early_stopping: bool = False
    use_multiprocessing: bool = True
    wandb_kwargs: dict = field(default_factory=dict)
    wandb_project: str = None
    warmup_ratio: float = 0.06
    warmup_steps: int = 0
    weight_decay: int = 0
    skip_special_tokens: bool = True

    def update_from_dict(self, new_values):
        """
        Update attributes of the ModelArgs instance from a dictionary.

        Args:
            new_values (dict): A dictionary containing attribute-value pairs to update.

        Raises:
            TypeError: If the input `new_values` is not a Python dictionary.

        Example:
            model_args = ModelArgs()
            new_values = {'learning_rate': 0.01, 'train_batch_size': 16}
            model_args.update_from_dict(new_values)
        """
        if isinstance(new_values, dict):
            for key, value in new_values.items():
                setattr(self, key, value)
        else:
            raise (TypeError(f"{new_values} is not a Python dict."))

    def get_args_for_saving(self):
        """
        Get a dictionary of model arguments suitable for saving.

        Returns:
            dict: A dictionary containing model arguments, excluding those specified in `not_saved_args`.

        Example:
            model_args = ModelArgs()
            args_to_save = model_args.get_args_for_saving()
        """
        args_for_saving = {
            key: value
            for key, value in asdict(self).items()
            if key not in self.not_saved_args
        }
        return args_for_saving

    def save(self, output_dir):
        """
        Save the model configuration to a JSON file in the specified output directory.

        Args:
            output_dir (str): The directory where the model configuration JSON file will be saved.

        Example:
            model_args = ModelArgs()
            model_args.save("output_directory")
        """
        os.makedirs(output_dir, exist_ok=True)
        with open(os.path.join(output_dir, "model_args.json"), "w") as f:
            json.dump(self.get_args_for_saving(), f)

    def load(self, input_dir):
        """
        Load the model configuration from a JSON file in the specified input directory.

        Args:
            input_dir (str): The directory where the model configuration JSON file is located.

        Example:
            model_args = ModelArgs()
            model_args.load("input_directory")
        """
        if input_dir:
            model_args_file = os.path.join(input_dir, "model_args.json")
            if os.path.isfile(model_args_file):
                with open(model_args_file, "r") as f:
                    model_args = json.load(f)

                self.update_from_dict(model_args)


@dataclass
class ClassificationArgs(ModelArgs):
    """
    Model args for a ClassificationModel
    """

    model_class: str = "ClassificationModel"
    """
    (str) The name of the classification model class. Defaults to "ClassificationModel".
    """

    labels_list: list = field(default_factory=list)
    """
    (list) A list of labels used for classification. Defaults to an empty list.
    """

    labels_map: dict = field(default_factory=dict)
    """
    (dict) A dictionary that maps labels to their corresponding indices. Defaults to an empty dictionary.
    """

    lazy_delimiter: str = "\t"
    """
    (str) The delimiter used for lazy loading of data. Defaults to the tab character ("\t").
    """

    lazy_labels_column: int = 1
    """
    (int) The column index (1-based) containing labels when using lazy loading. Defaults to 1.
    """

    lazy_loading: bool = False
    """
    (bool) Whether to use lazy loading of data. Defaults to False.
    """

    lazy_loading_start_line: int = 1
    """
    (int) The line number (1-based) to start reading data when using lazy loading. Defaults to 1.
    """

    lazy_text_a_column: bool = None
    """
    (bool) Whether the lazy loading data contains a text column for input "text_a". Defaults to None.
    """

    lazy_text_b_column: bool = None
    """
    (bool) Whether the lazy loading data contains a text column for input "text_b". Defaults to None.
    """

    lazy_text_column: int = 0
    """
    (int) The column index (0-based) containing text data when using lazy loading. Defaults to 0.
    """

    onnx: bool = False
    """
    (bool) Whether to use ONNX format for the model. Defaults to False.
    """

    regression: bool = False
    """
    (bool) Whether the task is regression (True) or classification (False). Defaults to False.
    """

    sliding_window: bool = False
    """
    (bool) Whether to use a sliding window approach for long documents. Defaults to False.
    """

    stride: float = 0.8
    """
    (float) The stride used in the sliding window approach. Defaults to 0.8.
    """

    tie_value: int = 1
    """
    (int) The value used for tied tokens in the dataset. Defaults to 1.
    """

    evaluate_during_training_steps: int = 20
    """
    (int) The number of steps between evaluations during training. Defaults to 20.
    """

    evaluate_during_training: bool = True
    """
    (bool) Whether to perform evaluations during training. Defaults to True.
    """


@dataclass
class SeqTaggingArgs(ModelArgs):
    """
    Model args for a SeqTaggingModel
    """

    model_class: str = "SeqTaggingModel"
    """
    (str) The name of the SeqTagging model class. Defaults to "SeqTaggingModel".
    """

    labels_list: list = field(default_factory=list)
    """
    (list) A list of labels used for sequence tagging. Defaults to an empty list.
    """

    lazy_delimiter: str = "\t"
    """
    (str) The delimiter used for lazy loading of data. Defaults to the tab character ("\t").
    """

    lazy_labels_column: int = 1
    """
    (int) The column index (1-based) containing labels when using lazy loading. Defaults to 1.
    """

    lazy_loading: bool = False
    """
    (bool) Whether to use lazy loading of data. Defaults to False.
    """

    lazy_loading_start_line: int = 1
    """
    (int) The line number (1-based) to start reading data when using lazy loading. Defaults to 1.
    """

    onnx: bool = False
    """
    (bool) Whether to use ONNX format for the model. Defaults to False.
    """

    evaluate_during_training_steps: int = 20
    """
    (int) The number of steps between evaluations during training. Defaults to 20.
    """

    evaluate_during_training: bool = True
    """
    (bool) Whether to perform evaluations during training. Defaults to True.
    """

    classification_report: bool = True
    """
    (bool) Whether to generate a classification report. Defaults to True.
    """

    pad_token_label_id: int = CrossEntropyLoss().ignore_index
    """
    (int) The ID of the pad token label used for padding. Defaults to CrossEntropyLoss().ignore_index.
    """


@dataclass
class SpanExtractionArgs(ModelArgs):
    """
    Model args for a SpanExtractionModel
    """

    model_class: str = "QuestionAnsweringModel"
    """
    (str) The name of the SpanExtraction model class. Defaults to "QuestionAnsweringModel".
    """

    doc_stride: int = 384
    """
    (int) The document stride for span extraction. Defaults to 384.
    """

    early_stopping_metric: str = "correct"
    """
    (str) The early stopping metric. Defaults to "correct".
    """

    early_stopping_metric_minimize: bool = False
    """
    (bool) Whether to minimize the early stopping metric. Defaults to False.
    """

    lazy_loading: bool = False
    """
    (bool) Whether to use lazy loading of data. Defaults to False.
    """

    max_answer_length: int = 100
    """
    (int) The maximum answer length. Defaults to 100.
    """

    max_query_length: int = 64
    """
    (int) The maximum query length. Defaults to 64.
    """

    n_best_size: int = 20
    """
    (int) The number of best answers to consider. Defaults to 20.
    """

    null_score_diff_threshold: float = 0.0
    """
    (float) The null score difference threshold. Defaults to 0.0.
    """

    evaluate_during_training_steps: int = 20
    """
    (int) The number of steps between evaluations during training. Defaults to 20.
    """

    evaluate_during_training: bool = True
    """
    (bool) Whether to perform evaluations during training. Defaults to True.
    """



@dataclass
class Seq2SeqArgs(ModelArgs):
    """
    Model args for a Seq2SeqModel
    """

    model_class: str = "Seq2SeqModel"
    """
    (str) The name of the Seq2Seq model class. Defaults to "Seq2SeqModel".
    """

    base_marian_model_name: str = None
    """
    (str) The base Marian model name. Defaults to None.
    """

    dataset_class: Dataset = None
    """
    (Dataset) The dataset class. Defaults to None.
    """

    do_sample: bool = False
    """
    (bool) Whether to perform sampling during decoding. Defaults to False.
    """

    early_stopping: bool = True
    """
    (bool) Whether to use early stopping during training. Defaults to True.
    """

    evaluate_generated_text: bool = False
    """
    (bool) Whether to evaluate generated text. Defaults to False.
    """

    length_penalty: float = 2.0
    """
    (float) The length penalty factor during decoding. Defaults to 2.0.
    """

    max_length: int = 20
    """
    (int) The maximum length of generated text. Defaults to 20.
    """

    max_steps: int = -1
    """
    (int) The maximum number of training steps. Defaults to -1 (unlimited).
    """

    num_beams: int = 4
    """
    (int) The number of beams used during decoding. Defaults to 4.
    """

    num_return_sequences: int = 1
    """
    (int) The number of generated sequences to return. Defaults to 1.
    """

    repetition_penalty: float = 1.0
    """
    (float) The repetition penalty factor during decoding. Defaults to 1.0.
    """

    top_k: float = None
    """
    (float) The top-k value used during decoding. Defaults to None.
    """

    top_p: float = None
    """
    (float) The top-p value used during decoding. Defaults to None.
    """

    use_multiprocessed_decoding: bool = False
    """
    (bool) Whether to use multiprocessed decoding. Defaults to False.
    """

    evaluate_during_training: bool = True
    """
    (bool) Whether to perform evaluations during training. Defaults to True.
    """

    src_lang: str = "en_XX"
    """
    (str) The source language for translation. Defaults to "en_XX".
    """

    tgt_lang: str = "ro_RO"
    """
    (str) The target language for translation. Defaults to "ro_RO".
    """

