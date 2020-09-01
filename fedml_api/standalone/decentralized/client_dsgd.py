import random

import torch


class ClientDSGD(object):
    def __init__(self, model, model_cache, client_id, streaming_data, topology_manager, iteration_number,
                 learning_rate, batch_size, weight_decay, latency, b_symmetric):
        # logging.info("streaming_data = %s" % streaming_data)

        # Since we use logistic regression, the model size is small.
        # Thus, independent model is created each client.
        self.model = model

        self.b_symmetric = b_symmetric
        self.topology_manager = topology_manager
        self.id = client_id  # integer
        self.streaming_data = streaming_data

        if self.b_symmetric:
            self.topology = topology_manager.get_symmetric_neighbor_list(client_id)
        else:
            self.topology = topology_manager.get_asymmetric_neighbor_list(client_id)
        # print(self.topology)

        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        self.criterion = torch.nn.BCELoss()

        self.learning_rate = learning_rate
        self.iteration_number = iteration_number
        # TODO:
        self.latency = random.uniform(0, latency)

        self.batch_size = batch_size
        self.loss_in_each_iteration = []

        # the default weight of the model is z_t, while the x weight is another weight used as temporary value
        self.model_x = model_cache

        # neighbors_weight_dict
        self.neighbors_weight_dict = dict()
        self.neighbors_topo_weight_dict = dict()

    def train_local(self, iteration_id):
        self.optimizer.zero_grad()
        train_x = torch.from_numpy(self.streaming_data[iteration_id]['x'])
        train_y = torch.FloatTensor([self.streaming_data[iteration_id]['y']])
        outputs = self.model(train_x)
        loss = self.criterion(outputs, train_y)
        loss.backward()
        self.optimizer.step()
        self.loss_in_each_iteration.append(loss)

    def train(self, iteration_id):
        self.optimizer.zero_grad()

        if iteration_id >= self.iteration_number:
            iteration_id = iteration_id % self.iteration_number

        train_x = torch.from_numpy(self.streaming_data[iteration_id]['x']).float()
        # print(train_x)
        train_y = torch.FloatTensor([self.streaming_data[iteration_id]['y']])
        outputs = self.model(train_x)
        # print(train_y)
        loss = self.criterion(outputs, train_y)
        grads_z = torch.autograd.grad(loss, self.model.parameters())

        for x_paras, g_z in zip(list(self.model_x.parameters()), grads_z):
            temp = g_z.data.mul(0 - self.learning_rate)
            x_paras.data.add_(temp)

        self.loss_in_each_iteration.append(loss)

    def get_regret(self):
        return self.loss_in_each_iteration

    # simulation
    def send_local_gradient_to_neighbor(self, client_list):
        for index in range(len(self.topology)):
            if self.topology[index] != 0 and index != self.id:
                client = client_list[index]
                client.receive_neighbor_gradients(self.id, self.model_x, self.topology[index])

    def receive_neighbor_gradients(self, client_id, model_x, topo_weight):
        self.neighbors_weight_dict[client_id] = model_x
        self.neighbors_topo_weight_dict[client_id] = topo_weight

    def update_local_parameters(self):
        # update x_{t+1/2}
        for x_paras in self.model_x.parameters():
            x_paras.data.mul_(self.topology[self.id])

        for client_id in self.neighbors_weight_dict.keys():
            model_x = self.neighbors_weight_dict[client_id]
            topo_weight = self.neighbors_topo_weight_dict[client_id]
            for x_paras, x_neighbor in zip(list(self.model_x.parameters()), list(model_x.parameters())):
                temp = x_neighbor.data.mul(topo_weight)
                x_paras.data.add_(temp)

        # update parameter z (self.model)
        for x_params, z_params in zip(list(self.model_x.parameters()), list(self.model.parameters())):
            z_params.data.copy_(x_params)
