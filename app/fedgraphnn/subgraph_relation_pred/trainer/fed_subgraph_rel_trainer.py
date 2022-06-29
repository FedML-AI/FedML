import logging

import numpy as np
import torch
import torch.nn.functional as F

import wandb

from torch_geometric.utils import negative_sampling

from sklearn.metrics import average_precision_score, roc_auc_score

from fedml.core.alg_frame.client_trainer import ClientTrainer

class FedSubgraphRelTrainer(ClientTrainer):

    def get_model_params(self):
        return self.model.cpu().state_dict()

    def set_model_params(self, model_parameters):
        logging.info("set_model_params")
        self.model.load_state_dict(model_parameters)

    def train(self, train_data, device, args):
        model = self.model

        self.metric_fn = average_precision_score if args.metric =="AP" else roc_auc_score

        model.to(device)
        model.train()

        val_data, test_data = None, None
        try:
            val_data = self.val_data
            test_data = self.test_data
        except:
            pass

        if args.client_optimizer == "sgd":
            optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=args.learning_rate, weight_decay = args.weight_decay)
        else:
            optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.learning_rate,  weight_decay = args.weight_decay)

        max_test_score = 0
        best_model_params = {}
        for epoch in range(args.epochs):

            for idx_batch, batch in enumerate(train_data):
                 optimizer.zero_grad()
                 z = model.encode(batch.edge_index, batch.edge_label)

                 pos_out = model.decode(z, batch.edge_index, batch.edge_label)

                 neg_edge_index = self.negative_sampling(batch.edge_index, batch.num_nodes)
                 neg_out = model.decode(z, neg_edge_index, batch.edge_label)



                 out = torch.cat([pos_out, neg_out])
                 gt = torch.cat([torch.ones_like(pos_out), torch.zeros_like(neg_out)])
                 cross_entropy_loss = F.binary_cross_entropy_with_logits(out, gt)
                 reg_loss = z.pow(2).mean() + model.decoder.rel_emb.pow(2).mean()
                 loss = cross_entropy_loss + 1e-2 * reg_loss
                 loss.backward()
                 optimizer.step()

            # if val_data:
            #     acc_v, _ = self.test(val_data, device)

            if ((idx_batch + 1) % args.frequency_of_the_test == 0) or (idx_batch == len(train_data) - 1):
                if test_data is not None:
                    test_score, _ = self.test(self.test_data, device)
                    print('Epoch = {}, Iter = {}/{}: Test accuracy = {}'.format(epoch, idx_batch + 1, len(train_data), test_score))
                    if test_score > max_test_score:
                        max_test_score = test_score
                        best_model_params = {k: v.cpu() for k, v in model.state_dict().items()}
                    print('Current best = {}'.format(max_test_score))

        return max_test_score, best_model_params

    def test(self, test_data, device):
        logging.info("----------test--------")
        model = self.model
        model.eval()
        model.to(device)

        cum_score = 0.
        ngraphs = 0
        for batch in test_data:
            batch.to(device)
            with torch.no_grad():
                pos_edge_index = batch['test_pos_edge_index']
                neg_edge_index = batch['test_neg_edge_index']
                link_logits = model.decode(self.train_z, pos_edge_index, neg_edge_index)
                link_probs = link_logits.sigmoid()
                link_labels = self.get_link_labels(pos_edge_index, neg_edge_index)
            cum_score += self.metric_fn(link_labels.cpu(), link_probs.cpu())
            ngraphs += batch.num_graphs

        return cum_score / ngraphs, model

    def test_on_the_server(self, train_data_local_dict, test_data_local_dict, device, args=None) -> bool:
        logging.info("----------test_on_the_server--------")

        model_list, score_list = [], []
        for client_idx in test_data_local_dict.keys():
            test_data = test_data_local_dict[client_idx]
            score, model = self.test(test_data, device)
            for idx in range(len(model_list)):
                self._compare_models(model, model_list[idx])
            model_list.append(model)
            score_list.append(score)
            logging.info('Client {}, Test {} = {}'.format(client_idx, args.metric, score))
            wandb.log({"Client {} Test/{}}".format(client_idx, args.metric): score})

        avg_score = np.mean(np.array(score_list))
        logging.info('Test {} = {}'.format(args.metric, avg_score))
        wandb.log({"Test/{}".format(args.metric) : avg_score})

        return True

    def _compare_models(self, model_1, model_2):
        models_differ = 0
        for key_item_1, key_item_2 in zip(model_1.state_dict().items(), model_2.state_dict().items()):
            if torch.equal(key_item_1[1], key_item_2[1]):
                pass
            else:
                models_differ += 1
                if key_item_1[0] == key_item_2[0]:
                    logging.info('Mismatch found at', key_item_1[0])
                else:
                    raise Exception
        if models_differ == 0:
            logging.info('Models match perfectly! :)')

    def negative_sampling(edge_index, num_nodes):
        # Sample edges by corrupting either the subject or the object of each edge.
        mask_1 = torch.rand(edge_index.size(1)) < 0.5
        mask_2 = ~mask_1

        neg_edge_index = edge_index.clone()
        neg_edge_index[0, mask_1] = torch.randint(num_nodes, (mask_1.sum(), ) , device = device)
        neg_edge_index[1, mask_2] = torch.randint(num_nodes, (mask_2.sum(), ), device = device)
        return neg_edge_index