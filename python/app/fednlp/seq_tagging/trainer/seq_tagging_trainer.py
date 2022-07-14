import copy
import os
import logging
import numpy as np
import torch
from seqeval.metrics import (
    f1_score,
    precision_score,
    recall_score,
)
from torch import nn
from tqdm import tqdm

from fedml.core import ClientTrainer
from fedml.data.fednlp.base.data_manager.base_data_manager import BaseDataManager
from fedml.model.nlp.model_args import SeqTaggingArgs
from .seq_tagging_utils import *


class MyModelTrainer(ClientTrainer):
    def get_model_params(self):
        return self.model.cpu().state_dict()

    def set_model_params(self, model_parameters):
        self.model.load_state_dict(model_parameters)

    def train(self, train_data, device, args, test_data=None):
        model_args = SeqTaggingArgs()
        model_args.model_name = args.model
        model_args.model_type = args.model_type
        # model_args.load(model_args.model_name)
        # model_args.num_labels = output_dim
        model_args.update_from_dict(
            {
                "fl_algorithm": args.federated_optimizer,
                "freeze_layers": args.freeze_layers,
                "epochs": args.epochs,
                "learning_rate": args.learning_rate,
                "gradient_accumulation_steps": args.gradient_accumulation_steps,
                "do_lower_case": args.do_lower_case,
                "manual_seed": args.random_seed,
                # for ignoring the cache features.
                "reprocess_input_data": args.reprocess_input_data,
                "overwrite_output_dir": True,
                "max_seq_length": args.max_seq_length,
                "train_batch_size": args.batch_size,
                "eval_batch_size": args.eval_batch_size,
                "evaluate_during_training": False,  # Disabled for FedAvg.
                "evaluate_during_training_steps": args.evaluate_during_training_steps,
                "fp16": args.fp16,
                "data_file_path": args.data_file_path,
                "partition_file_path": args.partition_file_path,
                "partition_method": args.partition_method,
                "dataset": args.dataset,
                "output_dir": args.output_dir,
                "is_debug_mode": args.is_debug_mode,
                "fedprox_mu": args.fedprox_mu,
                "optimizer": args.client_optimizer,
            }
        )
        model = self.model

        model.to(device)
        model.train()
        tr_loss = 0
        # train and update
        criterion = nn.CrossEntropyLoss().to(device)
        iteration_in_total = (
            len(train_data) // args.gradient_accumulation_steps * args.epochs
        )
        optimizer, scheduler = build_optimizer(model, iteration_in_total, model_args)
        if args.federated_optimizer == "FedProx":
            global_model = copy.deepcopy(model)
        epoch_loss = []
        for epoch in range(args.epochs):
            batch_loss = []
            for batch_idx, batch in tqdm(enumerate(train_data)):
                x = batch[1].to(device)
                labels = batch[4].to(device)
                log_probs = model(x)
                log_probs = log_probs[0]
                loss = criterion(log_probs.view(-1, args.num_labels), labels.view(-1))
                if args.federated_optimizer == "FedProx":
                    fed_prox_reg = 0.0
                    mu = args.fedprox_mu
                    for (p, g_p) in zip(model.parameters(), global_model.parameters()):
                        fed_prox_reg += (mu / 2) * torch.norm((p - g_p.data)) ** 2
                    loss += fed_prox_reg

                if args.gradient_accumulation_steps > 1:
                    loss = loss / args.gradient_accumulation_steps
                loss.backward()
                tr_loss += loss.item()
                if (batch_idx + 1) % args.gradient_accumulation_steps == 0:
                    if args.clip_grad_norm == 1:
                        torch.nn.utils.clip_grad_norm_(
                            model.parameters(), args.max_grad_norm
                        )
                    optimizer.step()
                    scheduler.step()  # Update learning rate schedule
                    model.zero_grad()
                    batch_loss.append(tr_loss)
                    tr_loss = 0
                    # if args.evaluate_during_training and (args.evaluate_during_training_steps > 0 and global_step % args.evaluate_during_training_steps == 0):
                    #    metrics = self.

                    # global_step += 1

                # Uncommet this following line to avoid nan loss
                # torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)

                # optimizer.step()

            epoch_loss.append(sum(batch_loss) / len(batch_loss))
            logging.info(
                "Client Index = {}\tEpoch: {}\tLoss: {:.6f}".format(
                    self.id, epoch, epoch_loss[-1]
                )
            )
            if args.evaluate_during_training and test_data is not None:
                results, _, _ = self.test(test_data, device, args)
                logging.info(
                    "Client Index = {}\tEpoch: {}\tF1 Score: {:.6f}".format(
                        self.id, epoch, results["f1_score"]
                    )
                )

    def test(self, test_data, device, args):
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

            end_index = (
                start_index + args.eval_batch_size
                if i != (n_batches - 1)
                else test_sample_len
            )

            if preds is None:
                preds = logits.detach().cpu().numpy()
                out_label_ids = batch[4].detach().cpu().numpy()
                out_input_ids = batch[1].detach().cpu().numpy()
                out_attention_mask = batch[2].detach().cpu().numpy()
            else:
                preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
                out_label_ids = np.append(
                    out_label_ids, batch[4].detach().cpu().numpy(), axis=0
                )
                out_input_ids = np.append(
                    out_input_ids, batch[1].detach().cpu().numpy(), axis=0
                )
                out_attention_mask = np.append(
                    out_attention_mask, batch[2].detach().cpu().numpy(), axis=0,
                )

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

    def test_on_the_server(
        self, train_data_local_dict, test_data_local_dict, device, args=None
    ) -> bool:
        logging.info("----------test_on_the_server--------")
        f1_list, metric_list = [], []
        for client_idx in test_data_local_dict.keys():
            test_data = test_data_local_dict[client_idx]
            metrics = self.test(test_data, device, args)
            metric_list.append(metrics)
            f1_list.append(metrics["f1_score"])
            logging.info(
                "Client {}, Test f1 = {}".format(client_idx, metrics["f1_score"])
            )
        avg_f1 = np.mean(np.array(f1_list))
        logging.info("Avg Test f1 = {}".format(avg_f1))
        return True
