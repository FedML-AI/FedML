from tqdm import tqdm

from fedml.data.fednlp.base.data_manager.base_data_manager import BaseDataManager


class SpanExtractionDataManager(BaseDataManager):
    """Data manager for reading comprehension (span-based QA)."""

    def __init__(self, args, model_args, preprocessor, process_id=0, num_workers=1):
        # TODO: ref to a defination of the "args" and "model_args"
        #           --- what will they must contain? (e.g., data_file_path)
        super(SpanExtractionDataManager, self).__init__(
            args, model_args, process_id, num_workers
        )
        self.attributes = self.load_attributes(args.data_file_path)
        self.preprocessor = preprocessor

    def read_instance_from_h5(self, data_file, index_list, desc=""):
        context_X = list()
        question_X = list()
        y = list()
        y_answers = list()
        qas_ids = list()
        for idx in tqdm(index_list, desc="Loading data from h5 file." + desc):
            context_X.append(data_file["context_X"][str(idx)][()].decode("utf-8"))
            question_X.append(data_file["question_X"][str(idx)][()].decode("utf-8"))
            y.append(data_file["Y"][str(idx)][()])
            y_answers.append(data_file["Y_answer"][str(idx)][()].decode("utf-8"))
            if "question_ids" in data_file:
                qas_ids.append(data_file["question_ids"][str(idx)][()].decode("utf-8"))
        return {
            "context_X": context_X,
            "question_X": question_X,
            "y": y,
            "y_answers": y_answers,
            "qas_ids": qas_ids if qas_ids else None,
        }
