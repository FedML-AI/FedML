import torch
from torch import nn

from ....core.alg_frame.client_trainer import ClientTrainer
from ....utils.compression import compressors
from ....utils.model_utils import (
    get_named_data,
    get_all_bn_params,
    check_device
)

import logging



class MyModelTrainer(ClientTrainer):
    def __init__(self, model, device, args=None):
        super().__init__(model, args)
        # =============================================
        self.device = device
        self.compressor = compressors[args.compression]()
        self.criterion = nn.CrossEntropyLoss().to(device)
        if args.client_optimizer == "sgd":
            self.optimizer = torch.optim.SGD(
                filter(lambda p: p.requires_grad, self.model.parameters()),
                lr=args.learning_rate, momentum=args.momentum
            )
        else:
            self.optimizer = torch.optim.Adam(
                filter(lambda p: p.requires_grad, self.model.parameters()),
                lr=args.learning_rate,
                weight_decay=args.weight_decay,
                amsgrad=True,
            )

    def get_model_params(self):
        # return 
        weights = self.model.cpu().state_dict()
        return weights
        # if self.args.compression is None or self.args.compression == 'no':
        #     compressed_weights = weights
        #     model_indexes = None

        # elif self.args.compression in ['topk', 'randomk', 'gtopk', 'randomkec', 'eftopk', 'gtopkef']:
        #     compressed_weights = {}
        #     model_indexes = {}
        #     for key in list(weights.keys()):
        #         logging.debug("weights[key].shape: {}, weights[key].numel(): {}".format(
        #             weights[key].shape, weights[key].numel()
        #         ))
        #         _, model_indexes[key], compressed_weights[key] = \
        #             self.compressor.compress(
        #                 self.compressor.flatten(weights[key]), name=key,
        #                 sigma_scale=3, ratio=self.args.compress_ratio
        #             )
        # elif self.args.compression in ['quantize', 'qsgd', 'sign']:
        #     compressed_weights = {}
        #     model_indexes = None
        #     for key in list(weights.keys()):
        #         logging.debug("weights[key].shape: {}, weights[key].numel(): {}".format(
        #             weights[key].shape, weights[key].numel()
        #         ))
        #         compressed_weights[key] = self.compressor.compress(
        #             weights[key], name=key,
        #             quantize_level=self.args.quantize_level, is_biased=self.args.is_biased
        #         )
        # else:
        #     raise NotImplementedError

        # return compressed_weights, model_indexes


    def set_model_params(self, model_parameters):
        self.model.load_state_dict(model_parameters)


    def get_model_grads(self):
        named_grads = get_named_data(self.model, mode='GRAD', use_cuda=True)

        # logging.debug(named_grads)
        if self.args.compression is None or self.args.compression == 'no':
            compressed_grads = named_grads
            grad_indexes = None
        elif self.args.compression in ['topk','randomk', 'eftopk']:
            compressed_grads = {}
            grad_indexes = {}
            for key in list(named_grads.keys()):
                logging.debug("named_grads[key].shape: {}, named_grads[key].numel(): {}".format(
                    named_grads[key].shape, named_grads[key].numel()
                ))
                _, grad_indexes[key], compressed_grads[key] = \
                    self.compressor.compress(
                        self.compressor.flatten(named_grads[key]), name=key,
                        sigma_scale=3, ratio=self.args.compress_ratio
                    )
        elif self.args.compression in ['quantize', 'qsgd', 'sign']:
            compressed_grads = {}
            grad_indexes = None
            for key in list(named_grads.keys()):
                logging.debug("named_grads[key].shape: {}, named_grads[key].numel(): {}".format(
                    named_grads[key].shape, named_grads[key].numel()
                ))
                compressed_grads[key] = self.compressor.compress(
                    named_grads[key], name=key,
                    quantize_level=self.args.quantize_level, is_biased=self.args.is_biased
                )
        else:
            raise NotImplementedError

        return compressed_grads, grad_indexes


    def set_grad_params(self, named_grads):
        # pass
        self.model.train()
        self.optimizer.zero_grad()
        for name, parameter in self.model.named_parameters():
            parameter.grad.copy_(named_grads[name].data.to(self.device))


    def clear_grad_params(self):
        self.optimizer.zero_grad()

    def get_model_bn(self):
        all_bn_params = get_all_bn_params(self.model)
        return all_bn_params

    def set_model_bn(self, all_bn_params):
        # logging.info(f"all_bn_params.keys(): {all_bn_params.keys()}")
        # for name, params in all_bn_params.items():
            # logging.info(f"name:{name}, params.shape: {params.shape}")
        for module_name, module in self.model.named_modules():
            if type(module) is nn.BatchNorm2d:
                # logging.info(f"module_name:{module_name}, params.norm: {module.weight.data.norm()}")
                module.weight.data = all_bn_params[module_name+".weight"] 
                module.bias.data = all_bn_params[module_name+".bias"] 
                module.running_mean = all_bn_params[module_name+".running_mean"] 
                module.running_var = all_bn_params[module_name+".running_var"] 
                module.num_batches_tracked = all_bn_params[module_name+".num_batches_tracked"] 


    def update_model_with_grad(self):
        self.model.to(self.device)
        self.optimizer.step()


    def infer_one_step(self, train_batch_data, device, args,
                    move_to_gpu, model_train, clear_grad_bef_opt):
        """
            inference and BP without optimization
        """
        model = self.model

        if move_to_gpu:
            model.to(device)

        if model_train:
            model.train()
        else:
            model.eval()

        x, labels = train_batch_data
        x, labels = x.to(device), labels.to(device)

        if clear_grad_bef_opt:
            # logging.info("zerograd!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
            # logging.info(f"dict(model.named_parameters())['conv1.weight'].grad[0]:\
            #     {dict(model.named_parameters())['conv1.weight'].grad[0]}")

            self.optimizer.zero_grad()
            # logging.info(f"dict(model.named_parameters())['conv1.weight'].grad:\
            #     {dict(model.named_parameters())['conv1.weight'].grad}")

        output = model(x)
        loss = self.criterion(output, labels)
        loss.backward()
        return loss.item()



    def train(self, train_data, device, args):
        model = self.model

        model.to(device)
        model.train()

        # train and update
        epoch_loss = []
        for epoch in range(args.epochs):
            batch_loss = []
            for batch_idx, (x, labels) in enumerate(train_data):
                x, labels = x.to(device), labels.to(device)
                model.zero_grad()
                log_probs = model(x)
                loss = self.criterion(log_probs, labels)
                loss.backward()

                # Uncommet this following line to avoid nan loss
                # torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)

                self.optimizer.step()
                logging.info(
                    "Update Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
                        epoch,
                        (batch_idx + 1) * args.batch_size,
                        len(train_data) * args.batch_size,
                        100.0 * (batch_idx + 1) / len(train_data),
                        loss.item(),
                    )
                )
                batch_loss.append(loss.item())
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

        metrics = {"test_correct": 0, "test_loss": 0, "test_total": 0}

        criterion = nn.CrossEntropyLoss().to(device)

        with torch.no_grad():
            for batch_idx, (x, target) in enumerate(test_data):
                x = x.to(device)
                target = target.to(device)
                pred = model(x)
                loss = criterion(pred, target)

                _, predicted = torch.max(pred, -1)
                correct = predicted.eq(target).sum()

                metrics["test_correct"] += correct.item()
                metrics["test_loss"] += loss.item() * target.size(0)
                metrics["test_total"] += target.size(0)
        return metrics

    def test_on_the_server(
        self, train_data_local_dict, test_data_local_dict, device, args=None
    ) -> bool:
        return False
