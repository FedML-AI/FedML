import json
from abc import ABC, abstractmethod

import h5py
import numpy as np
from tqdm import tqdm


class BaseRawDataLoader(ABC):
    """Abstract base class for raw data loaders.

    This class defines the common interface for raw data loaders, which are responsible for loading
    and processing raw data from various sources.

    Attributes:
        data_path (str): The path to the raw data.
        attributes (dict): A dictionary to store attributes related to the loaded data.

    Methods:
        load_data(): Abstract method to load the raw data.
        process_data_file(file_path): Abstract method to process a data file.
        generate_h5_file(file_path): Abstract method to generate an HDF5 file from the loaded data.

    """

    @abstractmethod
    def __init__(self, data_path):
        """Initialize the BaseRawDataLoader.

        Args:
            data_path (str): The path to the raw data.
        """
        self.data_path = data_path
        self.attributes = dict()
        self.attributes["index_list"] = None

    @abstractmethod
    def load_data(self):
        """Load the raw data.

        This method should be implemented by subclasses to load raw data from the specified data_path.

        Returns:
            None
        """
        pass

    @abstractmethod
    def process_data_file(self, file_path):
        """Process a data file.

        This method should be implemented by subclasses to process a specific data file.

        Args:
            file_path (str): The path to the data file to be processed.

        Returns:
            None
        """
        pass

    @abstractmethod
    def generate_h5_file(self, file_path):
        """Generate an HDF5 file from the loaded data.

        This method should be implemented by subclasses to generate an HDF5 file from the loaded data.

        Args:
            file_path (str): The path to the HDF5 file to be generated.

        Returns:
            None
        """
        pass


class TextClassificationRawDataLoader(BaseRawDataLoader):
    """Raw data loader for text classification tasks.

    This class extends the BaseRawDataLoader and provides specific functionality for loading and processing
    raw data for text classification tasks.

    Attributes:
        X (dict): A dictionary to store input data.
        Y (dict): A dictionary to store target labels.
        attributes (dict): Additional attributes related to the loaded data, including 'num_labels',
            'label_vocab', and 'task_type' which is set to "text_classification".

    Methods:
        generate_h5_file(file_path): Generate an HDF5 file from the loaded data.

    """

    def __init__(self, data_path):
        """Initialize the TextClassificationRawDataLoader.

        Args:
            data_path (str): The path to the raw data.
        """
        super(TextClassificationRawDataLoader, self).__init__(data_path)
        self.X = dict()
        self.Y = dict()
        self.attributes["num_labels"] = -1
        self.attributes["label_vocab"] = None
        self.attributes["task_type"] = "text_classification"

    def generate_h5_file(self, file_path):
        """Generate an HDF5 file from the loaded data.

        Args:
            file_path (str): The path to the HDF5 file to be generated.

        Returns:
            None
        """
        f = h5py.File(file_path, "w")
        f["attributes"] = json.dumps(self.attributes)
        for key in self.X.keys():
            f["X/" + str(key)] = self.X[key]
            f["Y/" + str(key)] = self.Y[key]
        f.close()


class SpanExtractionRawDataLoader(BaseRawDataLoader):
    """Raw data loader for span extraction tasks.

    This class extends the BaseRawDataLoader and provides specific functionality for loading and processing
    raw data for span extraction tasks.

    Attributes:
        context_X (dict): A dictionary to store context input data.
        question_X (dict): A dictionary to store question input data.
        Y (dict): A dictionary to store target spans.
        Y_answer (dict): A dictionary to store target answers.
        attributes (dict): Additional attributes related to the loaded data, including 'task_type' which is
            set to "span_extraction".

    Methods:
        generate_h5_file(file_path): Generate an HDF5 file from the loaded data.

    """

    def __init__(self, data_path):
        """Initialize the SpanExtractionRawDataLoader.

        Args:
            data_path (str): The path to the raw data.
        """
        super(SpanExtractionRawDataLoader, self).__init__(data_path)
        self.context_X = dict()
        self.question_X = dict()
        self.Y = dict()
        self.attributes["task_type"] = "span_extraction"
        self.Y_answer = dict()

    def generate_h5_file(self, file_path):
        """Generate an HDF5 file from the loaded data.

        Args:
            file_path (str): The path to the HDF5 file to be generated.

        Returns:
            None
        """
        f = h5py.File(file_path, "w")
        f["attributes"] = json.dumps(self.attributes)
        for key in self.context_X.keys():
            f["context_X/" + str(key)] = self.context_X[key]
            f["question_X/" + str(key)] = self.question_X[key]
            f["Y/" + str(key)] = self.Y[key]
            f["Y_answer/" + str(key)] = self.Y_answer[key]
        f.close()


class SeqTaggingRawDataLoader(BaseRawDataLoader):
    """Raw data loader for sequence tagging tasks.

    This class extends the BaseRawDataLoader and provides specific functionality for loading and processing
    raw data for sequence tagging tasks.

    Attributes:
        X (dict): A dictionary to store input sequences.
        Y (dict): A dictionary to store target labels.
        attributes (dict): Additional attributes related to the loaded data, including 'num_labels',
            'label_vocab', and 'task_type' which is set to "seq_tagging".

    Methods:
        generate_h5_file(file_path): Generate an HDF5 file from the loaded data.

    """

    def __init__(self, data_path):
        """Initialize the SeqTaggingRawDataLoader.

        Args:
            data_path (str): The path to the raw data.
        """
        super(SeqTaggingRawDataLoader, self).__init__(data_path)
        self.X = dict()
        self.Y = dict()
        self.attributes["num_labels"] = -1
        self.attributes["label_vocab"] = None
        self.attributes["task_type"] = "seq_tagging"

    def generate_h5_file(self, file_path):
        """Generate an HDF5 file from the loaded data.

        Args:
            file_path (str): The path to the HDF5 file to be generated.

        Returns:
            None
        """
        f = h5py.File(file_path, "w")
        f["attributes"] = json.dumps(self.attributes)
        utf8_type = h5py.string_dtype("utf-8", None)
        for key in self.X.keys():
            f["X/" + str(key)] = np.array(self.X[key], dtype=utf8_type)
            f["Y/" + str(key)] = np.array(self.Y[key], dtype=utf8_type)
        f.close()


class Seq2SeqRawDataLoader(BaseRawDataLoader):
    """Raw data loader for sequence-to-sequence (seq2seq) tasks.

    This class extends the BaseRawDataLoader and provides specific functionality for loading and processing
    raw data for sequence-to-sequence tasks.

    Attributes:
        X (dict): A dictionary to store source sequences.
        Y (dict): A dictionary to store target sequences.
        task_type (str): The type of the task, which is set to "seq2seq".

    Methods:
        generate_h5_file(file_path): Generate an HDF5 file from the loaded data.

    """

    def __init__(self, data_path):
        """Initialize the Seq2SeqRawDataLoader.

        Args:
            data_path (str): The path to the raw data.
        """
        super(Seq2SeqRawDataLoader, self).__init__(data_path)
        self.X = dict()
        self.Y = dict()
        self.task_type = "seq2seq"

    def generate_h5_file(self, file_path):
        """Generate an HDF5 file from the loaded data.

        Args:
            file_path (str): The path to the HDF5 file to be generated.

        Returns:
            None
        """
        f = h5py.File(file_path, "w")
        f["attributes"] = json.dumps(self.attributes)
        for key in self.X.keys():
            f["X/" + str(key)] = self.X[key]
            f["Y/" + str(key)] = self.Y[key]
        f.close()


class LanguageModelRawDataLoader(BaseRawDataLoader):
    """Raw data loader for language modeling tasks.

    This class extends the BaseRawDataLoader and provides specific functionality for loading and processing
    raw data for language modeling tasks.

    Attributes:
        X (dict): A dictionary to store language model input data.
        task_type (str): The type of the task, which is set to "lm".

    Methods:
        generate_h5_file(file_path): Generate an HDF5 file from the loaded data.

    """

    def __init__(self, data_path):
        """Initialize the LanguageModelRawDataLoader.

        Args:
            data_path (str): The path to the raw data.
        """
        super(LanguageModelRawDataLoader, self).__init__(data_path)
        self.X = dict()
        self.task_type = "lm"

    def generate_h5_file(self, file_path):
        """Generate an HDF5 file from the loaded data.

        Args:
            file_path (str): The path to the HDF5 file to be generated.

        Returns:
            None
        """
        f = h5py.File(file_path, "w")
        f["attributes"] = json.dumps(self.attributes)
        for key in tqdm(self.X.keys(), desc="generate data h5 file"):
            f["X/" + str(key)] = self.X[key]
        f.close()
