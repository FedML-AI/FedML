import torch
from torch import nn

# from ....core.alg_frame.client_trainer import ClientTrainer
from fedml.core.alg_frame.client_trainer import ClientTrainer

from fedml.data.reddit.nlp import mask_tokens

import logging


class RedditTrainer(ClientTrainer):
    from transformers import (AdamW, AlbertTokenizer, AutoConfig,
                              AutoModelWithLMHead, AutoTokenizer,
                              MobileBertForPreTraining)

    tokenizer = AlbertTokenizer.from_pretrained(
        'albert-base-v2', do_lower_case=True)

    def get_model_params(self):
        return self.model.cpu().state_dict()

    def set_model_params(self, model_parameters):
        self.model.load_state_dict(model_parameters)


    def create_optimizer(self, model):

        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": self.args.weight_decay,
            },
            {
                "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        # Bert pre-training setup
        optimizer = torch.optim.Adam(
            optimizer_grouped_parameters, lr=self.args.learning_rate, weight_decay=1e-2)        
        return optimizer


    def train(self, train_data, device, args):
        model = self.model

        model.to(device)
        model.train()

        # train and update
        criterion = nn.CrossEntropyLoss().to(device)
        optimizer = self.create_optimizer(model)

        epoch_loss = []
        for epoch in range(args.epochs):
            batch_loss = []
            for batch_idx, (x, _) in enumerate(train_data):
                x, labels = mask_tokens(
                    x, self.tokenizer, self.args, device=device)

                x, labels = x.to(device), labels.to(device)
                model.zero_grad()
                outputs = model(x, labels=labels)
                loss = outputs[0]
                loss_list = [loss.item()]  # [loss.mean().data.item()]
                loss.backward()
                # Uncommet this following line to avoid nan loss
                # torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)

                optimizer.step()
  
                temp_loss = sum(loss_list)/float(len(loss_list))
                batch_loss.append(temp_loss)
            if len(batch_loss) == 0:
                epoch_loss.append(0.0)
            else:
                epoch_loss.append(sum(batch_loss) / len(batch_loss))
            logging.info(
                "Client Index = {}\tEpoch: {}\tLoss: {:.6f}".format(
                    self.id, epoch, sum(epoch_loss) / len(epoch_loss)
                )
            )




    def test(self, test_data, device, args):
        model = self.model

        model.to(device)
        model.eval()

        metrics = {"test_correct": 0, "test_loss": 0, "test_total": 0, "perplexity_loss": 0.}

        criterion = nn.CrossEntropyLoss().to(device)

        with torch.no_grad():
            for batch_idx, (x, _) in enumerate(test_data):
                # if args.mlm else (data, data)
                x, target = mask_tokens(
                    x, self.tokenizer, args, device=device)
                x = x.to(device)
                target = target.to(device)
                pred = model(x)
                loss = criterion(pred, target)

                outputs = model(x, labels=target)

                loss = outputs[0]
                test_loss += loss.data.item()
                perplexity_loss += loss.data.item()

                acc = accuracy(
                    outputs[1].reshape(-1, outputs[1].shape[2]), target.reshape(-1), topk=(1, 5))

                correct += acc[0].item()
                top_5 += acc[1].item()

                _, predicted = torch.max(pred, -1)
                correct = predicted.eq(target).sum()

                metrics["test_correct"] += correct
                metrics["test_loss"] += test_loss * target.size(0)
                metrics["test_total"] += target.size(0)
                metrics["perplexity_loss"] += perplexity_loss * target.size(0)
        return metrics


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.reshape(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k)

        return res







