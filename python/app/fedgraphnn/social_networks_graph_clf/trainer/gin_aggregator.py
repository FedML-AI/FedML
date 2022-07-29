import logging

import numpy as np
import torch
import wandb

from fedml.core import ServerAggregator


class GINSocialNetworkAggregator(ServerAggregator):
    def get_model_params(self):
        return self.model.cpu().state_dict()

    def set_model_params(self, model_parameters):
        logging.info("set_model_params")
        self.model.load_state_dict(model_parameters)

    def test(self, test_data, device, args):
        pass

    def _test(self, test_data, device, args):
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

    def test_all(self, train_data_local_dict, test_data_local_dict, device, args=None) -> bool:
        logging.info("----------test_on_the_server--------")

        model_list, score_list = [], []
        for client_idx in test_data_local_dict.keys():
            test_data = test_data_local_dict[client_idx]
            score, model = self._test(test_data, device, args)
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
        for key_item_1, key_item_2 in zip(model_1.state_dict().items(), model_2.state_dict().items()):
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