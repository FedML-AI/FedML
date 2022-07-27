import os

import numpy as np
import torch
from seqeval.metrics import (
    f1_score,
    precision_score,
    recall_score,
)
from torch import nn

from fedml.core import ServerAggregator
from fedml.data.fednlp.base.data_manager.base_data_manager import BaseDataManager
from .seq_tagging_utils import *


# Trainer for MoleculeNet. The evaluation metric is ROC-AUC


class TaggingAggregator(ServerAggregator):
    def get_model_params(self):
        return self.model.cpu().state_dict()

    def set_model_params(self, model_parameters):
        logging.info("set_model_params")
        self.model.load_state_dict(model_parameters)

    def test(self, test_data, device, args):
        pass

    def _test(self, test_data, device):
        args = self.args
        attributes = BaseDataManager.load_attributes(args.data_file_path)
        args.num_labels = len(attributes["label_vocab"])
        args.labels_list = list(attributes["label_vocab"].keys())
        args.pad_token_label_id = nn.CrossEntropyLoss().ignore_index
        results = {}
        eval_loss = 0.0
        nb_eval_steps = 0

        n_batches = len(test_data)

        test_sample_len = len(test_data.dataset)
        pad_token_label_id = args.pad_token_label_id
        eval_output_dir = args.output_dir

        preds = None
        out_label_ids = None

        self.model.to(device)
        self.model.eval()
        # logging.info("len(test_dl) = %d, n_batches = %d" % (test_data, n_batches))
        for i, batch in enumerate(test_data):
            batch = tuple(t for t in batch)
            with torch.no_grad():
                sample_index_list = batch[0].to(device).cpu().numpy()

                x = batch[1].to(device)
                labels = batch[4].to(device)

                output = self.model(x)
                logits = output[0]

                loss_fct = nn.CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, args.num_labels), labels.view(-1))
                eval_loss += loss.item()
                # logging.info("test. batch index = %d, loss = %s" % (i, str(eval_loss)))

            nb_eval_steps += 1
            start_index = args.eval_batch_size * i

            end_index = start_index + args.eval_batch_size if i != (n_batches - 1) else test_sample_len

            if preds is None:
                preds = logits.detach().cpu().numpy()
                out_label_ids = batch[4].detach().cpu().numpy()
                out_input_ids = batch[1].detach().cpu().numpy()
                out_attention_mask = batch[2].detach().cpu().numpy()
            else:
                preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
                out_label_ids = np.append(out_label_ids, batch[4].detach().cpu().numpy(), axis=0)
                out_input_ids = np.append(out_input_ids, batch[1].detach().cpu().numpy(), axis=0)
                out_attention_mask = np.append(out_attention_mask, batch[2].detach().cpu().numpy(), axis=0,)

        eval_loss = eval_loss / nb_eval_steps

        token_logits = preds
        preds = np.argmax(preds, axis=2)

        label_map = {i: label for i, label in enumerate(args.labels_list)}

        out_label_list = [[] for _ in range(out_label_ids.shape[0])]
        preds_list = [[] for _ in range(out_label_ids.shape[0])]

        for i in range(out_label_ids.shape[0]):
            for j in range(out_label_ids.shape[1]):
                if out_label_ids[i, j] != pad_token_label_id:
                    out_label_list[i].append(label_map[out_label_ids[i][j]])
                    preds_list[i].append(label_map[preds[i][j]])
        # logging.info(preds_list[:2])
        # logging.info(out_label_list[:2])
        result = {
            "eval_loss": eval_loss,
            "precision": precision_score(out_label_list, preds_list),
            "recall": recall_score(out_label_list, preds_list),
            "f1_score": f1_score(out_label_list, preds_list),
        }
        # logging.info(result)

        os.makedirs(eval_output_dir, exist_ok=True)
        output_eval_file = os.path.join(eval_output_dir, "eval_results.txt")
        with open(output_eval_file, "w") as writer:
            for key in sorted(result.keys()):
                writer.write("{} = {}\n".format(key, str(result[key])))

        # self.results.update(result)

        return result

    def test_all(self, train_data_local_dict, test_data_local_dict, device, args) -> bool:
        logging.info("----------test_on_the_server--------")
        f1_list, metric_list = [], []
        for client_idx in test_data_local_dict.keys():
            test_data = test_data_local_dict[client_idx]
            metrics = self.test(test_data, device, args)
            metric_list.append(metrics)
            f1_list.append(metrics["f1_score"])
            logging.info("Client {}, Test f1 = {}".format(client_idx, metrics["f1_score"]))
        avg_f1 = np.mean(np.array(f1_list))
        logging.info("Avg Test f1 = {}".format(avg_f1))
        return True
