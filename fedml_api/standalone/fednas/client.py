import torch
from torch import nn

from darts import utils
from darts.architect import Architect


class Client:

    def __init__(self, client_idx, local_training_data, local_test_data, local_sample_number, args,
                 logger, device, is_wandb_used=False):
        self.client_idx = client_idx
        self.local_training_data = local_training_data
        self.local_test_data = local_test_data

        self.local_sample_number = local_sample_number

        self.criterion = nn.CrossEntropyLoss().to(device)
        self.args = args
        self.logger = logger

        self.device = device
        self.is_wandb_used = is_wandb_used

    def get_sample_number(self):
        return self.local_sample_number

    def local_search(self, net):
        net.train()

        arch_parameters = net.arch_parameters()
        arch_params = list(map(id, arch_parameters))

        parameters = net.parameters()
        weight_params = filter(lambda p: id(p) not in arch_params,
                               parameters)

        optimizer = torch.optim.SGD(
            weight_params,  # model.parameters(),
            self.args.learning_rate,
            momentum=self.args.momentum,
            weight_decay=self.args.weight_decay)

        architect = Architect(net, self.criterion, self.args, self.device)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, float(self.args.epochs), eta_min=self.args.learning_rate_min)

        local_avg_train_acc = []
        for epoch in range(self.args.epochs):
            # training
            train_acc, train_obj, train_loss = self.search(self.local_training_data, self.local_test_data,
                                                           net, architect, self.criterion,
                                                           optimizer)
            self.logger.info('client_idx = %d, local search_acc %f' % (self.client_idx, train_acc))
            local_avg_train_acc.append(train_acc)

            # # validation
            # with torch.no_grad():
            #     valid_acc, valid_obj, valid_loss = self.infer(self.local_test_data, net, self.criterion)
            # self.logger.info('client_idx = %d, local valid_acc %f' % (self.client_idx, valid_acc))

            scheduler.step()
            lr = scheduler.get_lr()[0]
            self.logger.info('client_idx = %d, epoch %d lr %e' % (self.client_idx, epoch, lr))

        return net.state_dict(), net.arch_parameters(), sum(local_avg_train_acc) / len(local_avg_train_acc)

    def local_train(self, net):
        net.train()

        parameters = net.parameters()

        optimizer = torch.optim.SGD(
            parameters,  # model.parameters(),
            self.args.learning_rate,
            momentum=self.args.momentum,
            weight_decay=self.args.weight_decay)

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, float(self.args.epochs), eta_min=self.args.learning_rate_min)

        local_avg_train_acc = []
        for epoch in range(self.args.epochs):
            # training
            train_acc, train_obj, train_loss = self.train(self.local_training_data, self.local_test_data,
                                                          net, self.criterion,
                                                          optimizer)
            self.logger.info('client_idx = %d, local train_acc %f' % (self.client_idx, train_acc))
            local_avg_train_acc.append(train_acc)

            scheduler.step()
            lr = scheduler.get_lr()[0]
            self.logger.info('client_idx = %d, epoch %d lr %e' % (self.client_idx, epoch, lr))

        return net.state_dict(), sum(local_avg_train_acc) / len(local_avg_train_acc)

    def search(self, train_queue, valid_queue, model, architect, criterion, optimizer):
        objs = utils.AvgrageMeter()
        top1 = utils.AvgrageMeter()
        top5 = utils.AvgrageMeter()

        for step, (input, target) in enumerate(train_queue):

            # logging.info("epoch %d, step %d START" % (epoch, step))
            model.train()
            n = input.size(0)

            input = input.to(self.device)
            target = target.to(self.device)

            # get a random minibatch from the search queue with replacement
            input_search, target_search = next(iter(valid_queue))
            input_search = input_search.to(self.device)
            target_search = target_search.to(self.device)

            architect.step_v2(input, target, input_search, target_search, self.args.lambda_train_regularizer,
                              self.args.lambda_valid_regularizer)

            optimizer.zero_grad()
            logits = model(input)
            loss = criterion(logits, target)

            loss.backward()
            parameters = model.arch_parameters()
            nn.utils.clip_grad_norm_(parameters, self.args.grad_clip)
            optimizer.step()
            # logging.info("step %d. update weight by SGD. FINISH\n" % step)

            prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
            objs.update(loss.item(), n)
            top1.update(prec1.item(), n)
            top5.update(prec5.item(), n)

            # torch.cuda.empty_cache()

            if step % self.args.report_freq == 0:
                self.logger.info('search %03d %e %f %f', step, objs.avg, top1.avg, top5.avg)

        return top1.avg, objs.avg, loss

    def train(self, train_queue, valid_queue, model, criterion, optimizer):
        objs = utils.AvgrageMeter()
        top1 = utils.AvgrageMeter()
        top5 = utils.AvgrageMeter()

        for step, (input, target) in enumerate(train_queue):
            # logging.info("epoch %d, step %d START" % (epoch, step))
            model.train()
            n = input.size(0)

            input = input.to(self.device)
            target = target.to(self.device)

            optimizer.zero_grad()
            logits, logits_aux = model(input)
            loss = criterion(logits, target)
            if self.args.auxiliary:
                loss_aux = criterion(logits_aux, target)
                loss += self.args.auxiliary_weight * loss_aux
            loss.backward()
            parameters = model.parameters()
            nn.utils.clip_grad_norm_(parameters, self.args.grad_clip)
            optimizer.step()
            # logging.info("step %d. update weight by SGD. FINISH\n" % step)

            prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
            objs.update(loss.item(), n)
            top1.update(prec1.item(), n)
            top5.update(prec5.item(), n)

            # torch.cuda.empty_cache()

            if step % self.args.report_freq == 0:
                self.logger.info('train %03d %e %f %f', step, objs.avg, top1.avg, top5.avg)

        return top1.avg, objs.avg, loss

    def infer(self, valid_queue, model, criterion):

        objs = utils.AvgrageMeter()
        top1 = utils.AvgrageMeter()
        top5 = utils.AvgrageMeter()
        model.eval()

        for step, (input, target) in enumerate(valid_queue):
            input = input.to(self.device)
            target = target.to(self.device)

            logits = model(input)
            loss = criterion(logits, target)

            prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
            n = input.size(0)
            objs.update(loss.item(), n)
            top1.update(prec1.item(), n)
            top5.update(prec5.item(), n)

            if step % self.args.report_freq == 0:
                self.logger.info('valid %03d %e %f %f', step, objs.avg, top1.avg, top5.avg)

        return top1.avg, objs.avg, loss

    def local_test(self, model_global, b_use_test_dataset=False):
        model_global.eval()
        model_global.to(self.device)
        test_loss = test_acc = test_total = 0.
        if b_use_test_dataset:
            test_data = self.local_test_data
        else:
            test_data = self.local_training_data
        with torch.no_grad():
            for batch_idx, (x, target) in enumerate(test_data):
                x = x.to(self.device)
                target = target.to(self.device)

                pred = model_global(x)
                loss = self.criterion(pred, target)
                _, predicted = torch.max(pred, 1)
                correct = predicted.eq(target).sum()

                test_acc += correct.item()
                test_loss += loss.item() * target.size(0)
                test_total += target.size(0)

        return test_acc, test_total, test_loss
