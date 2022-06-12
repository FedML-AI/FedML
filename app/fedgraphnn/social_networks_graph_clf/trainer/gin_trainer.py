import logging

import numpy as np
import torch
import wandb

from fedml.core.alg_frame.client_trainer import ClientTrainer


class GINSocialNetworkTrainer(ClientTrainer):
    def get_model_params(self):
        return self.model.cpu().state_dict()

    def set_model_params(self, model_parameters):
        logging.info("set_model_params")
        self.model.load_state_dict(model_parameters)

    def train(self, train_data, device, args):
        model = self.model

        model.to(device)
        model.train()

        val_data, test_data = None, None
        try:
            val_data = self.val_data
            test_data = self.test_data
        except:
            pass

        if args.client_optimizer == "sgd":
            optimizer = torch.optim.SGD(
                filter(lambda p: p.requires_grad, model.parameters()),
                lr=args.learning_rate,
                weight_decay=args.weight_decay,
            )
        else:
            optimizer = torch.optim.Adam(
                filter(lambda p: p.requires_grad, model.parameters()),
                lr=args.learning_rate,
                weight_decay=args.weight_decay,
            )

        max_test_acc = 0
        best_model_params = {}
        for epoch in range(args.epochs):
            ngraphs = 0
            acc_sum = 0

            for idx_batch, batch in enumerate(train_data):
                batch.to(device)
                optimizer.zero_grad()
                pred = model(batch)
                label = batch.y
                acc_sum += pred.max(dim=1)[1].eq(label).sum().item()
                loss = model.loss(pred, label)
                loss.backward()
                optimizer.step()
                ngraphs += batch.num_graphs
            acc = acc_sum / ngraphs

            # if val_data:
            #     acc_v, _ = self.test(val_data, device)

            if ((idx_batch + 1) % args.frequency_of_the_test == 0) or (
                idx_batch == len(train_data) - 1
            ):
                if test_data is not None:
                    test_acc, _ = self.test(self.test_data, device)
                    print(
                        "Epoch = {}, Iter = {}/{}: Test accuracy = {}".format(
                            epoch, idx_batch + 1, len(train_data), acc
                        )
                    )
                    if test_acc > max_test_acc:
                        max_test_acc = test_acc
                        best_model_params = {
                            k: v.cpu() for k, v in model.state_dict().items()
                        }
                    print("Current best = {}".format(max_test_acc))

        return max_test_acc, best_model_params

    def test(self, test_data, device):
        logging.info("----------test--------")
        model = self.model
        model.eval()
        model.to(device)

        acc_sum = 0.0
        ngraphs = 0
        for batch in test_data:
            batch.to(device)
            with torch.no_grad():
                pred = model(batch)
                label = batch.y
            acc_sum += pred.max(dim=1)[1].eq(label).sum().item()
            ngraphs += batch.num_graphs

        return acc_sum / ngraphs, model

    def test_on_the_server(
        self, train_data_local_dict, test_data_local_dict, device, args=None
    ) -> bool:
        logging.info("----------test_on_the_server--------")

        model_list, score_list = [], []
        for client_idx in test_data_local_dict.keys():
            test_data = test_data_local_dict[client_idx]
            score, model = self.test(test_data, device)
            # for idx in range(len(model_list)):
            #     self._compare_models(model, model_list[idx])
            model_list.append(model)
            score_list.append(score)
            logging.info("Client {}, Test accuracy = {}".format(client_idx, score))
            if args.enable_wandb:
                wandb.log({"Client {} Test/accuracy".format(client_idx): score})

        avg_score = np.mean(np.array(score_list))
        logging.info("Test accuracy = {}".format(avg_score))
        if args.enable_wandb:
            wandb.log({"Test/accuracy": avg_score})

        return True

    def _compare_models(self, model_1, model_2):
        models_differ = 0
        for key_item_1, key_item_2 in zip(
            model_1.state_dict().items(), model_2.state_dict().items()
        ):
            if torch.equal(key_item_1[1], key_item_2[1]):
                pass
            else:
                models_differ += 1
                if key_item_1[0] == key_item_2[0]:
                    logging.info("Mismatch found at", key_item_1[0])
                else:
                    raise Exception
        if models_differ == 0:
            logging.info("Models match perfectly! :)")
