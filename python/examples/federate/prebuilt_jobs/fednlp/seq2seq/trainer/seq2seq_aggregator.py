import os
from multiprocessing import Pool

import torch
from tqdm import tqdm

from fedml.core import ServerAggregator
from .seq2seq_utils import *


class Seq2SeqAggregator(ServerAggregator):
    def get_model_params(self):
        return self.model.cpu().state_dict()

    def set_model_params(self, model_parameters):
        logging.info("set_model_params")
        self.model.load_state_dict(model_parameters)

    def test(self, test_data, device, args):
        pass

    def _test(self, test_data, device):
        args = self.args

        results = {}

        eval_loss = 0.0
        rouge_score = 0.0

        # bluert_score = 0.0
        # bluert_checkpoint = "~/fednlp_data/bleurt-base-128"
        # bleurt_scorer = bleurt.score.BleurtScorer(bluert_checkpoint)

        nb_eval_steps = 0

        n_batches = len(test_data)

        test_sample_len = len(test_data.dataset)
        # pad_token_label_id = self.pad_token_label_id
        eval_output_dir = self.args.output_dir

        preds = None
        out_label_ids = None

        self.model.to(device)
        self.model.eval()
        for i, batch in enumerate(test_data):
            # batch = tuple(t for t in batch)
            inputs = self._get_inputs_dict(batch, device)
            with torch.no_grad():
                outputs = self.model(**inputs)
                tmp_eval_loss = outputs[0]
                summary_ids = self.model.generate(
                    inputs["input_ids"],
                    num_beams=self.args.num_beams,
                    max_length=self.args.max_length,
                    early_stopping=True,
                )
                hyp_list = [
                    self.decoder_tokenizer.decode(
                        g, skip_special_tokens=True, clean_up_tokenization_spaces=False
                    ).strip()
                    for g in summary_ids
                ]
                ref_list = [
                    self.decoder_tokenizer.decode(
                        g, skip_special_tokens=True, clean_up_tokenization_spaces=False
                    ).strip()
                    for g in inputs["decoder_input_ids"]
                ]
                rouge = Rouge()
                refs = {idx: [line] for (idx, line) in enumerate(ref_list)}
                hyps = {idx: [line] for (idx, line) in enumerate(hyp_list)}
                res = rouge.compute_score(refs, hyps)
                rouge_score += res[0]
                # logits = output[0]
                # loss_fct = CrossEntropyLoss()
                # loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
                eval_loss += tmp_eval_loss.item()
                # logging.info("test. batch index = %d, loss = %s" % (i, str(eval_loss)))

            nb_eval_steps += 1
            start_index = self.args.eval_batch_size * i

            end_index = start_index + self.args.eval_batch_size if i != (n_batches - 1) else test_sample_len
        #   logging.info(
        #      "batch index = %d, start_index = %d, end_index = %d"
        #     % (i, start_index, end_index)
        # )

        eval_loss = eval_loss / nb_eval_steps
        rouge_score = rouge_score / nb_eval_steps

        result = {"eval_loss": eval_loss, "rouge_score": rouge_score}

        results.update(result)

        os.makedirs(eval_output_dir, exist_ok=True)
        output_eval_file = os.path.join(eval_output_dir, "eval_results.txt")
        with open(output_eval_file, "w") as writer:
            for key in sorted(result.keys()):
                writer.write("{} = {}\n".format(key, str(result[key])))
        self.results.update(result)

        model_preds = None

        # if self.args.evaluate_generated_text:
        # to_predict = [ex.input_text for ex in self.test_dl.examples]
        # references = [ex.target_text for ex in self.test_dl.examples]
        # model_preds = self.predict(to_predict)

        # TODO: compute ROUGE/BLUE/ scores here.
        # result = self.compute_metrics(references, model_preds)
        # self.results.update(result)

        # logging.info(self.results)

        return result, model_preds, None

    def test_all(self, train_data_local_dict, test_data_local_dict, device, args) -> bool:
        logging.info("----------test_on_the_server--------")
        f1_list, metric_list = [], []
        for client_idx in test_data_local_dict.keys():
            test_data = test_data_local_dict[client_idx]
            metrics, _, _ = self.test(test_data, device, args)
            metric_list.append(metrics)
            f1_list.append(metrics["rouge_score"])
            logging.info("Client {}, Test rouge_score = {}".format(client_idx, metrics["rouge_score"]))
        avg_accuracy = np.mean(np.array(f1_list))
        logging.info("Test avg rouge_score = {}".format(avg_accuracy))
        return True

    def _get_inputs_dict(self, batch, device):
        # device = self.device
        if self.args.model_type in ["bart", "marian"]:
            pad_token_id = self.encoder_tokenizer.pad_token_id
            source_ids, source_mask, y = (
                batch["source_ids"],
                batch["source_mask"],
                batch["target_ids"],
            )
            y_ids = y[:, :-1].contiguous()
            lm_labels = y[:, 1:].clone()
            lm_labels[y[:, 1:] == pad_token_id] = -100

            inputs = {
                "input_ids": source_ids.to(device),
                "attention_mask": source_mask.to(device),
                "decoder_input_ids": y_ids.to(device),
                "labels": lm_labels.to(device),
            }
        elif self.args.model_type in ["mbart"]:
            inputs = {
                "input_ids": batch["input_ids"].to(device),
                "attention_mask": batch["attention_mask"].to(device),
                "decoder_input_ids": batch["decoder_input_ids"].to(device),
                "labels": batch["labels"].to(device),
            }
        else:
            lm_labels = batch[1]
            lm_labels_masked = lm_labels.clone()
            lm_labels_masked[lm_labels_masked == self.decoder_tokenizer.pad_token_id] = -100

            inputs = {
                "input_ids": batch[0].to(device),
                "decoder_input_ids": lm_labels.to(device),
                "labels": lm_labels_masked.to(device),
            }

        return inputs

    def predict(self, to_predict, device):
        """
        Performs predictions on a list of text.
        Args:
            to_predict: A python list of text (str) to be sent to the model for prediction. Note that the prefix should be prepended to the text.
        Returns:
            preds: A python list of the generated sequences.
        """  # noqa: ignore flake8"

        # self._move_model_to_device()

        all_outputs = []
        # Batching
        for batch in [
            to_predict[i : i + self.args.eval_batch_size] for i in range(0, len(to_predict), self.args.eval_batch_size)
        ]:
            if self.args.model_type == "marian":
                input_ids = self.encoder_tokenizer.prepare_seq2seq_batch(
                    batch,
                    max_length=self.args.max_seq_length,
                    padding="max_length",
                    return_tensors="pt",
                    truncation=True,
                )["input_ids"]
            elif self.args.model_type in ["mbart"]:
                input_ids = self.encoder_tokenizer.prepare_seq2seq_batch(
                    src_texts=batch,
                    max_length=self.args.max_seq_length,
                    pad_to_max_length=True,
                    padding="max_length",
                    truncation=True,
                    src_lang=self.args.src_lang,
                )["input_ids"]
            else:
                input_ids = self.encoder_tokenizer.batch_encode_plus(
                    batch,
                    max_length=self.args.max_seq_length,
                    padding="max_length",
                    return_tensors="pt",
                    truncation=True,
                )["input_ids"]
            input_ids = input_ids.to(device)

            if self.args.model_type in ["bart", "marian"]:
                outputs = self.model.generate(
                    input_ids=input_ids,
                    num_beams=self.args.num_beams,
                    max_length=self.args.max_length,
                    length_penalty=self.args.length_penalty,
                    early_stopping=self.args.early_stopping,
                    repetition_penalty=self.args.repetition_penalty,
                    do_sample=self.args.do_sample,
                    top_k=self.args.top_k,
                    top_p=self.args.top_p,
                    num_return_sequences=self.args.num_return_sequences,
                )
            elif self.args.model_type in ["mbart"]:
                tgt_lang_token = self.decoder_tokenizer._convert_token_to_id(self.args.tgt_lang)

                outputs = self.model.generate(
                    input_ids=input_ids,
                    decoder_start_token_id=tgt_lang_token,
                    num_beams=self.args.num_beams,
                    max_length=self.args.max_length,
                    length_penalty=self.args.length_penalty,
                    early_stopping=self.args.early_stopping,
                    repetition_penalty=self.args.repetition_penalty,
                    do_sample=self.args.do_sample,
                    top_k=self.args.top_k,
                    top_p=self.args.top_p,
                    num_return_sequences=self.args.num_return_sequences,
                )
            else:
                outputs = self.model.generate(
                    input_ids=input_ids,
                    decoder_start_token_id=self.model.config.decoder.pad_token_id,
                    num_beams=self.args.num_beams,
                    max_length=self.args.max_length,
                    length_penalty=self.args.length_penalty,
                    early_stopping=self.args.early_stopping,
                    repetition_penalty=self.args.repetition_penalty,
                    do_sample=self.args.do_sample,
                    top_k=self.args.top_k,
                    top_p=self.args.top_p,
                    num_return_sequences=self.args.num_return_sequences,
                )

            all_outputs.extend(outputs.cpu().numpy())

        if self.args.use_multiprocessed_decoding:
            self.model.to("cpu")
            with Pool(self.args.process_count) as p:
                outputs = list(
                    tqdm(
                        p.imap(self._decode, all_outputs, chunksize=self.args.multiprocessing_chunksize,),
                        total=len(all_outputs),
                        desc="Decoding outputs",
                        disable=self.args.silent,
                    )
                )
            self._move_model_to_device()
        else:
            outputs = [
                self.decoder_tokenizer.decode(
                    output_id, skip_special_tokens=self.args.skip_special_tokens, clean_up_tokenization_spaces=True,
                )
                for output_id in all_outputs
            ]

        if self.args.num_return_sequences > 1:
            return [
                outputs[i : i + self.args.num_return_sequences]
                for i in range(0, len(outputs), self.args.num_return_sequences)
            ]
        else:
            return outputs

    def _decode(self, output_id):
        return self.decoder_tokenizer.decode(
            output_id, skip_special_tokens=self.args.skip_special_tokens, clean_up_tokenization_spaces=True,
        )
