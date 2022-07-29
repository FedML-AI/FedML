import logging

import numpy as np
import torch
import wandb
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
)
from sklearn.metrics import average_precision_score, roc_auc_score
from torch_geometric.utils import negative_sampling
from fedml.core import ServerAggregator


class FedLPAggregator(ServerAggregator):
    def get_model_params(self):
        return self.model.cpu().state_dict()

    def set_model_params(self, model_parameters):
        logging.info("set_model_params")
        self.model.load_state_dict(model_parameters)

    def test(self, test_data, device, args):
        pass

    def _test(self, test_data, device, val=True, metric=mean_absolute_error):
        logging.info("----------test--------")
        model = self.model
        model.eval()
        model.to(device)
        
        cum_score = 0.0
        ngraphs = 0
        threshold = torch.tensor([0.7], device = device)
        for batch in test_data:
            batch.to(device)
            with torch.no_grad():
               
                neg_edge_index = negative_sampling(
                    edge_index=batch.edge_index,
                    num_nodes=batch.num_nodes,
                    num_neg_samples=batch.edge_index.size(1),
                )
                z = model.encode(batch.x, batch.edge_index)
                out = model.decode(z, batch.edge_index, neg_edge_index).view(-1).sigmoid()
                pred = (out > threshold).float() * 1
            
            cum_score += average_precision_score(np.ones(pred.numel()), pred.cpu())
            print(cum_score)
            ngraphs += batch.num_graphs

        return cum_score / ngraphs, model

    def test_all(self, train_data_local_dict, test_data_local_dict, device, args) -> bool:
        logging.info("----------test_on_the_server--------")

        model_list, score_list, mae_list, rmse_list, mse_list = [], [], [], [], []
        for client_idx in test_data_local_dict.keys():
            test_data = test_data_local_dict[client_idx]
            score, model = self._test(test_data, device, val=False)

            for idx in range(len(model_list)):
                self._compare_models(model, model_list[idx])
            model_list.append(model)
            score_list.append(score)


            logging.info(
                "Client {}, Test {} = {}".format(
                    client_idx, args.metric, score
                )
            )
            if args.enable_wandb:
                wandb.log(
                    {"Client {} Test/{}".format(client_idx, args.metric): score,}
                )

        avg_score = np.mean(np.array(score_list))


        logging.info(
            "Test {} = {}".format(
                args.metric, avg_score
            )
        )
        if args.enable_wandb:
            wandb.log(
                {
                    "Client {} Test/{}".format(client_idx, args.metric): avg_score,
                }
            )

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

    def get_link_labels(self, pos_edge_index, neg_edge_index, device):
        num_links = pos_edge_index.size(1) + neg_edge_index.size(1)
        link_labels = torch.zeros(num_links, dtype=torch.float, device=device)
        link_labels[: pos_edge_index.size(1)] = 1.0
        return link_labels
