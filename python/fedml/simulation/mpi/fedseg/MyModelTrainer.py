import logging
import time

import numpy as np
import torch

from .utils import SegmentationLosses, Evaluator, LR_Scheduler, EvaluationMetricsKeeper
from ....core.alg_frame.client_trainer import ClientTrainer


class MyModelTrainer(ClientTrainer):
    def get_model_params(self):
        if self.args.backbone_freezed:
            logging.info("Initializing model; Backbone Freezed")
            return self.model.encoder_decoder.cpu().state_dict()
        else:
            logging.info("Initializing end-to-end model")
            return self.model.cpu().state_dict()

    def set_model_params(self, model_parameters):
        if self.args.backbone_freezed:
            logging.info("Updating Global model; Backbone Freezed")
            self.model.encoder_decoder.load_state_dict(model_parameters)
        else:
            logging.info("Updating Global model")
            self.model.load_state_dict(model_parameters)

    def train(self, train_data, device, args):
        model = self.model
        args = self.args

        model.to(device)
        model.train()

        criterion = SegmentationLosses().build_loss(mode=args.loss_type)
        scheduler = LR_Scheduler(
            args.lr_scheduler, args.lr, args.epochs, len(train_data)
        )

        if args.client_optimizer == "sgd":

            if args.backbone_freezed:
                optimizer = torch.optim.SGD(
                    filter(lambda p: p.requires_grad, self.model.parameters()),
                    lr=args.lr * 10,
                    momentum=args.momentum,
                    weight_decay=args.weight_decay,
                    nesterov=args.nesterov,
                )
            else:
                train_params = [
                    {"params": self.model.get_1x_lr_params(), "lr": args.lr},
                    {"params": self.model.get_10x_lr_params(), "lr": args.lr * 10},
                ]

                optimizer = torch.optim.SGD(
                    train_params,
                    momentum=args.momentum,
                    weight_decay=args.weight_decay,
                    nesterov=args.nesterov,
                )
        else:
            optimizer = torch.optim.Adam(
                filter(lambda p: p.requires_grad, self.model.parameters()),
                lr=args.lr,
                weight_decay=args.weight_decay,
                amsgrad=True,
            )

        epoch_loss = []

        for epoch in range(args.epochs):
            t = time.time()
            batch_loss = []
            logging.info("Trainer_ID: {0}, Epoch: {1}".format(self.id, epoch))

            for (batch_idx, batch) in enumerate(train_data):
                x, labels = batch["image"], batch["label"]
                x, labels = x.to(device), labels.to(device)
                scheduler(optimizer, batch_idx, epoch)
                optimizer.zero_grad()
                log_probs = model(x)
                loss = criterion(log_probs, labels).to(device)  # pylint: disable=E1102
                loss.backward()
                optimizer.step()
                batch_loss.append(loss.item())
                if batch_idx % 100 == 0:
                    logging.info(
                        "Trainer_ID: {0} Iteration: {1}, Loss: {2}, Time Elapsed: {3}".format(
                            self.id, batch_idx, loss, (time.time() - t) / 60
                        )
                    )

            if len(batch_loss) > 0:
                epoch_loss.append(sum(batch_loss) / len(batch_loss))
                logging.info(
                    "(Trainer_ID: {}. Local Training Epoch: {} \tLoss: {:.6f}".format(
                        self.id, epoch, sum(epoch_loss) / len(epoch_loss)
                    )
                )

    def test(self, test_data, device, args):
        logging.info("Evaluation on trainer ID:{}".format(self.id))
        model = self.model
        args = self.args
        evaluator = Evaluator(model.n_classes)

        model.eval()
        model.to(device)

        t = time.time()
        evaluator.reset()
        test_acc = (
            test_acc_class
        ) = test_mIoU = test_FWIoU = test_loss = test_total = 0.0
        criterion = SegmentationLosses().build_loss(mode=args.loss_type)

        with torch.no_grad():
            for (batch_idx, batch) in enumerate(test_data):
                x, target = batch["image"], batch["label"]
                x, target = x.to(device), target.to(device)
                output = model(x)
                loss = criterion(output, target).to(device)  # pylint: disable=E1102
                test_loss += loss.item()
                test_total += target.size(0)
                pred = output.cpu().numpy()
                target = target.cpu().numpy()
                pred = np.argmax(pred, axis=1)
                evaluator.add_batch(target, pred)
                if batch_idx % 100 == 0:
                    logging.info(
                        "Trainer_ID: {0} Iteration: {1}, Loss: {2}, Time Elapsed: {3}".format(
                            self.id, batch_idx, loss, (time.time() - t) / 60
                        )
                    )

                # time_end_test_per_batch = time.time()
                # logging.info("time per batch = " + str(time_end_test_per_batch - time_start_test_per_batch))
                # logging.info("Client = {0} Batch = {1}".format(self.client_index, batch_idx)

        # Evaluation Metrics (Averaged over number of samples)
        test_acc = evaluator.Pixel_Accuracy()
        test_acc_class = evaluator.Pixel_Accuracy_Class()
        test_mIoU = evaluator.Mean_Intersection_over_Union()
        test_FWIoU = evaluator.Frequency_Weighted_Intersection_over_Union()
        test_loss = test_loss / test_total

        logging.info(
            "Trainer_ID={0}, test_acc={1}, test_acc_class={2}, test_mIoU={3}, test_FWIoU={4}, test_loss={5}".format(
                self.id, test_acc, test_acc_class, test_mIoU, test_FWIoU, test_loss
            )
        )

        eval_metrics = EvaluationMetricsKeeper(
            test_acc, test_acc_class, test_mIoU, test_FWIoU, test_loss
        )
        return eval_metrics
