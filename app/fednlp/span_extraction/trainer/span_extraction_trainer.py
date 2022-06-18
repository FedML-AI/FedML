import torch
from torch import nn

from fedml.core import ClientTrainer
import logging
from .span_extraction_utils import *

import copy
import logging
import math
import os
from ...data.data_manager.base_data_manager import BaseDataManager
from ...model_args import SeqTaggingArgs
import numpy as np
import sklearn
import wandb
from transformers import (
    AdamW,
    get_linear_schedule_with_warmup,
)
from tqdm import tqdm


class MyModelTrainer(ClientTrainer):
    def __init__(
        self, args, device, model, train_dl=None, test_dl=None, tokenizer=None
    ):

        super(MyModelTrainer, self).__init__(model, args=args)
        self.device = device
        self.tokenizer = tokenizer
        self.results = {}

    def get_model_params(self):
        return self.model.cpu().state_dict()

    def set_model_params(self, model_parameters):
        self.model.load_state_dict(model_parameters)

    def train(self, train_data, device, args, test_data=None):
        if not device:
            device = self.device
        logging.info("train_model self.device: " + str(device))
        self.model.to(device)

        args = self.args

        no_decay = ["bias", "LayerNorm.weight"]

        optimizer_grouped_parameters = []
        custom_parameter_names = set()
        for group in args.custom_parameter_groups:
            params = group.pop("params")
            custom_parameter_names.update(params)
            param_group = {**group}
            param_group["params"] = [
                p for n, p in self.model.named_parameters() if n in params
            ]
            optimizer_grouped_parameters.append(param_group)

        for group in args.custom_layer_parameters:
            layer_number = group.pop("layer")
            layer = f"layer.{layer_number}."
            group_d = {**group}
            group_nd = {**group}
            group_nd["weight_decay"] = 0.0
            params_d = []
            params_nd = []
            for n, p in self.model.named_parameters():
                if n not in custom_parameter_names and layer in n:
                    if any(nd in n for nd in no_decay):
                        params_nd.append(p)
                    else:
                        params_d.append(p)
                    custom_parameter_names.add(n)
            group_d["params"] = params_d
            group_nd["params"] = params_nd

            optimizer_grouped_parameters.append(group_d)
            optimizer_grouped_parameters.append(group_nd)

        if not args.train_custom_parameters_only:
            optimizer_grouped_parameters.extend(
                [
                    {
                        "params": [
                            p
                            for n, p in self.model.named_parameters()
                            if n not in custom_parameter_names
                            and not any(nd in n for nd in no_decay)
                        ],
                        "weight_decay": args.weight_decay,
                    },
                    {
                        "params": [
                            p
                            for n, p in self.model.named_parameters()
                            if n not in custom_parameter_names
                            and any(nd in n for nd in no_decay)
                        ],
                        "weight_decay": 0.0,
                    },
                ]
            )

        # build optimizer and scheduler
        iteration_in_total = (
            len(train_data) // args.gradient_accumulation_steps * args.epochs
        )
        optimizer, scheduler = build_optimizer(self.model, iteration_in_total, args)

        if args.n_gpu > 1:
            print("gpu number", args.n_gpu)
            logging.info("torch.nn.DataParallel(self.model)")
            self.model = torch.nn.DataParallel(self.model)

        # training result
        global_step = 0
        training_progress_scores = None
        tr_loss, logging_loss = 0.0, 0.0
        self.model.zero_grad()
        best_eval_metric = None
        early_stopping_counter = 0
        steps_trained_in_current_epoch = 0
        epochs_trained = 0

        if args.evaluate_during_training:
            training_progress_scores = self._create_training_progress_scores()

        if args.fp16:
            from torch.cuda import amp

            scaler = amp.GradScaler()

        if self.args.fl_algorithm == "FedProx":
            global_model = copy.deepcopy(self.model)
        epoch_loss = []
        for epoch in range(0, args.epochs):
            batch_loss = []
            for batch_idx, batch in tqdm(enumerate(train_data)):
                self.model.train()
                batch = tuple(t.to(device) for t in batch)
                # dataset = TensorDataset(all_guid, all_input_ids, all_attention_masks, all_token_type_ids, all_cls_index,
                # all_p_mask, all_is_impossible)

                inputs = self._get_inputs_dict(batch)

                if args.fp16:
                    with amp.autocast():
                        outputs = self.model(**inputs)
                        # model outputs are always tuple in pytorch-transformers (see doc)
                        loss = outputs[0]
                else:
                    outputs = self.model(**inputs)
                    # model outputs are always tuple in pytorch-transformers (see doc)
                    loss = outputs[0]

                if args.n_gpu > 1:
                    loss = (
                        loss.mean()
                    )  # mean() to average on multi-gpu parallel training

                if self.args.fl_algorithm == "FedProx":
                    fed_prox_reg = 0.0
                    mu = self.args.fedprox_mu
                    for (p, g_p) in zip(
                        self.model.parameters(), global_model.parameters()
                    ):
                        fed_prox_reg += (mu / 2) * torch.norm((p - g_p.data)) ** 2
                    loss += fed_prox_reg

                current_loss = loss.item()

                if args.gradient_accumulation_steps > 1:
                    loss = loss / args.gradient_accumulation_steps
                if args.fp16:
                    scaler.scale(loss).backward()
                else:
                    loss.backward()
                tr_loss += loss.item()

                if (batch_idx + 1) % args.gradient_accumulation_steps == 0:
                    if args.fp16:
                        scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), args.max_grad_norm
                    )

                    if args.fp16:
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        optimizer.step()
                    scheduler.step()  # Update learning rate schedule
                    self.model.zero_grad()
                    global_step += 1
                    batch_loss.append(tr_loss)
                    tr_loss = 0
            epoch_loss.append(sum(batch_loss) / len(batch_loss))
            logging.info(
                "Client Index = {}\tEpoch: {}\tLoss: {:.6f}".format(
                    self.id, epoch, sum(epoch_loss) / len(epoch_loss)
                )
            )
            if (
                self.args.evaluate_during_training
                and (
                    self.args.evaluate_during_training_steps > 0
                    and global_step % self.args.evaluate_during_training_steps == 0
                )
                and test_data is not None
            ):
                results, _, _ = self.test(test_data, device, args)
                logging.info(results)

    def test(self, test_data, device, args):
        output_dir = self.args.output_dir

        if not device:
            device = self.device

        logging.info("test_model self.device: " + str(device))
        self.model.to(device)

        all_predictions, all_nbest_json, scores_diff_json, eval_loss = self.evaluate(
            output_dir, test_data, device
        )

        result, texts = self.calculate_results(all_predictions, test_data)
        result["eval_loss"] = eval_loss

        logging.info(result)

        return result, all_predictions, texts["incorrect_text"]

    def evaluate(self, output_dir, test_data, device, verbose_logging=False):
        """
        Evaluates the model on eval_data.
        Utility function to be used by the eval_model() method. Not intended to be used directly.
        """
        tokenizer = self.tokenizer
        # device = torch.device("cuda:0")
        model = self.model
        model.to(device)
        args = self.args

        # # reassgin unique_id for features to keep order for federated learning situation
        # unique_id = 1000000000
        # for feature in self.test_dl.features:
        #     feature.unique_id = unique_id
        #     unique_id += 1

        examples = test_data.examples
        features = test_data.features

        eval_loss = 0.0
        nb_eval_steps = 0
        model.eval()

        # if args.n_gpu > 1:
        #     model = torch.nn.DataParallel(model)

        if self.args.fp16:
            from torch.cuda import amp

        all_results = []
        for batch in tqdm(test_data, disable=args.silent, desc="Running Evaluation"):
            batch = tuple(t.to(device) for t in batch)

            with torch.no_grad():
                inputs = {
                    "input_ids": batch[1],
                    "attention_mask": batch[2],
                    "token_type_ids": batch[3],
                }

                if self.args.model_type in [
                    "xlm",
                    "roberta",
                    "distilbert",
                    "camembert",
                    "electra",
                    "xlmroberta",
                    "bart",
                ]:
                    del inputs["token_type_ids"]

                example_indices = batch[4]

                if args.model_type in ["xlnet", "xlm"]:
                    inputs.update({"cls_index": batch[5], "p_mask": batch[6]})

                if self.args.fp16:
                    with amp.autocast():
                        outputs = model(**inputs)
                        eval_loss += outputs[0].mean().item()
                else:
                    outputs = model(**inputs)
                    eval_loss += outputs[0].mean().item()
                begin_idx = len(all_results)
                for i, _ in enumerate(example_indices):
                    eval_feature = features[begin_idx + i]
                    unique_id = int(eval_feature.unique_id)
                    if args.model_type in ["xlnet", "xlm"]:
                        # XLNet uses a more complex post-processing procedure
                        result = RawResultExtended(
                            unique_id=unique_id,
                            start_top_log_probs=to_list(outputs[0][i]),
                            start_top_index=to_list(outputs[1][i]),
                            end_top_log_probs=to_list(outputs[2][i]),
                            end_top_index=to_list(outputs[3][i]),
                            cls_logits=to_list(outputs[4][i]),
                        )
                    else:
                        result = RawResult(
                            unique_id=unique_id,
                            start_logits=to_list(outputs[0][i]),
                            end_logits=to_list(outputs[1][i]),
                        )
                    all_results.append(result)

            nb_eval_steps += 1

        eval_loss = eval_loss / nb_eval_steps

        prefix = "test"
        os.makedirs(output_dir, exist_ok=True)

        output_prediction_file = os.path.join(
            output_dir, "predictions_{}.json".format(prefix)
        )
        output_nbest_file = os.path.join(
            output_dir, "nbest_predictions_{}.json".format(prefix)
        )
        output_null_log_odds_file = os.path.join(
            output_dir, "null_odds_{}.json".format(prefix)
        )

        if args.model_type in ["xlnet", "xlm"]:
            # XLNet uses a more complex post-processing procedure
            (
                all_predictions,
                all_nbest_json,
                scores_diff_json,
                out_eval,
            ) = write_predictions_extended(
                examples,
                features,
                all_results,
                args.n_best_size,
                args.max_answer_length,
                output_prediction_file,
                output_nbest_file,
                output_null_log_odds_file,
                None,
                model.config.start_n_top,
                model.config.end_n_top,
                True,
                tokenizer,
                verbose_logging,
            )
        else:
            all_predictions, all_nbest_json, scores_diff_json = write_predictions(
                examples,
                features,
                all_results,
                args.n_best_size,
                args.max_answer_length,
                args.do_lower_case,
                output_prediction_file,
                output_nbest_file,
                output_null_log_odds_file,
                verbose_logging,
                True,
                args.null_score_diff_threshold,
            )

        return all_predictions, all_nbest_json, scores_diff_json, eval_loss

    def calculate_results(self, predictions, test_data, **kwargs):
        # implement FedNLP evaluate function
        all_examples = test_data.examples
        exact_raw, f1_raw = get_raw_scores(all_examples, predictions)

        exact_dict = {}
        f1_dict = {}
        counter_dict = {}
        text_dict = {}
        for example in all_examples:
            guid = example.guid
            pred = predictions[guid]
            qid = example.qas_id
            if qid not in exact_dict:
                exact_dict[qid] = 0
            if qid not in f1_dict:
                f1_dict[qid] = 0
            exact_dict[qid] = max(exact_dict[qid], exact_raw[guid])
            f1_dict[qid] = max(f1_dict[qid], f1_raw[guid])
            answer = example.answer_text
            if answer.strip() == pred.strip():
                counter_dict[qid] = 2
                text_dict[qid] = {
                    "truth": answer,
                    "predicted": pred,
                    "question": example.question_text,
                }
            elif answer.strip() in pred.strip() or pred.strip() in answer.strip():
                if qid not in counter_dict or counter_dict[qid] < 1:
                    counter_dict[qid] = 1
                    text_dict[qid] = {
                        "truth": answer,
                        "predicted": pred,
                        "question": example.question_text,
                    }
            else:
                if qid not in counter_dict:
                    counter_dict[qid] = 0
                    text_dict[qid] = {
                        "truth": answer,
                        "predicted": pred,
                        "question": example.question_text,
                    }

        correct_text = {}
        similar_text = {}
        incorrect_text = {}
        correct = 0
        similar = 0
        incorrect = 0
        for qid, val in counter_dict.items():
            if val == 2:
                correct_text[qid] = text_dict[qid]
                correct += 1
            elif val == 1:
                similar_text[qid] = text_dict[qid]
                similar += 1
            else:
                incorrect_text[qid] = text_dict[qid]
                incorrect += 1

        standard_metrics = {
            "exact_match": sum(exact_raw.values()) / len(exact_raw),
            "f1_score": sum(f1_raw.values()) / len(f1_raw),
        }

        result = {
            "correct": correct,
            "similar": similar,
            "incorrect": incorrect,
            **standard_metrics,
        }
        # wandb.log(result)

        texts = {
            "correct_text": correct_text,
            "similar_text": similar_text,
            "incorrect_text": incorrect_text,
        }

        return result, texts

    def test_on_the_server(
        self, train_data_local_dict, test_data_local_dict, device, args=None
    ) -> bool:
        logging.info("----------test_on_the_server--------")
        f1_list, metric_list = [], []
        for client_idx in test_data_local_dict.keys():
            test_data = test_data_local_dict[client_idx]
            metrics, _, _ = self.test(test_data, device, args)
            metric_list.append(metrics)
            f1_list.append(metrics["f1_score"])
            logging.info(
                "Client {}, Test F1 = {}".format(client_idx, metrics["f1_score"])
            )
        avg_accuracy = np.mean(np.array(f1_list))
        logging.info("Test avg F1 = {}".format(avg_accuracy))
        return True

    def _create_training_progress_scores(self, **kwargs):
        extra_metrics = {key: [] for key in kwargs}
        training_progress_scores = {
            "global_step": [],
            "correct": [],
            "similar": [],
            "incorrect": [],
            "train_loss": [],
            "eval_loss": [],
            **extra_metrics,
        }

        return training_progress_scores

    def _get_inputs_dict(self, batch):
        inputs = {
            "input_ids": batch[1],
            "attention_mask": batch[2],
            "token_type_ids": batch[3],
            "start_positions": batch[4],
            "end_positions": batch[5],
        }

        if self.args.model_type in [
            "xlm",
            "roberta",
            "distilbert",
            "camembert",
            "electra",
            "xlmroberta",
            "bart",
        ]:
            del inputs["token_type_ids"]

        if self.args.model_type in ["xlnet", "xlm"]:
            inputs.update({"cls_index": batch[6], "p_mask": batch[7]})

        return inputs
