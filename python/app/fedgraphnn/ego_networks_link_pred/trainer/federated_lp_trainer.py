import logging

import numpy as np
import torch
import torch.nn.functional as F
import wandb
from sklearn.metrics import average_precision_score, roc_auc_score
from torch_geometric.utils import negative_sampling

from fedml.core.alg_frame.client_trainer import ClientTrainer


class FedLinkPredTrainer(ClientTrainer):
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

        max_test_score = 0
        best_model_params = {}
        for epoch in range(args.epochs):
            ngraphs = 0
            cum_score = 0

            for idx_batch, batch in enumerate(train_data):
                neg_edge_index = negative_sampling(
                    edge_index=batch.edge_index,
                    num_nodes=batch.num_nodes,
                    num_neg_samples=batch.edge_index.size(1),
                )

                batch.to(device)
                optimizer.zero_grad()


                z = model.encode(batch.x, batch.edge_index)
                self.train_z = z

                edge_idx, neg_idx  = batch.edge_index.to(device) , neg_edge_index.to(device)

                link_logits = model.decode(
                    z, edge_idx, neg_idx
                )
                link_labels = self.get_link_labels(
                    edge_idx, neg_idx, device
                )
                loss = F.binary_cross_entropy_with_logits(link_logits, link_labels)
                loss.backward()
                optimizer.step()

            # if val_data:
            #     acc_v, _ = self.test(val_data, device)

            if ((idx_batch + 1) % args.frequency_of_the_test == 0) or (
                idx_batch == len(train_data) - 1
            ):
                if test_data is not None:
                    test_score, _ = self.test(self.test_data, device)
                    print(
                        "Epoch = {}, Iter = {}/{}: Test accuracy = {}".format(
                            epoch, idx_batch + 1, len(train_data), test_score
                        )
                    )
                    if test_score > max_test_score:
                        max_test_score = test_score
                        best_model_params = {
                            k: v.cpu() for k, v in model.state_dict().items()
                        }
                    print("Current best = {}".format(max_test_score))

        return max_test_score, best_model_params

    def test(self, test_data, device, args):
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
            
            cum_score += average_precision_score(np.ones(batch.edge_index.numel()), pred.cpu())
            ngraphs += batch.num_graphs

        return cum_score / ngraphs, model

    def get_link_labels(self, pos_edge_index, neg_edge_index, device):
        num_links = pos_edge_index.size(1) + neg_edge_index.size(1)
        link_labels = torch.zeros(num_links, dtype=torch.float, device=device)
        link_labels[: pos_edge_index.size(1)] = 1.0
        return link_labels
