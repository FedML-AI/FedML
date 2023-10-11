import random

import torch


class ClientDSGD(object):
    """
    Client for Distributed Stochastic Gradient Descent (DSGD).

    Args:
        model: The machine learning model used by the client.
        model_cache: The model cache used for temporary values.
        client_id (int): The unique identifier of the client.
        streaming_data (list): Streaming data for training.
        topology_manager: The manager for defining communication topology.
        iteration_number (int): The total number of iterations.
        learning_rate (float): The learning rate for gradient descent.
        batch_size (int): The batch size for training.
        weight_decay (float): The weight decay for regularization.
        latency (float): The communication latency.
        b_symmetric (bool): Flag for symmetric or asymmetric communication topology.

    Methods:
        train_local(iteration_id):
            Train the client's model on local data for a specified iteration.
        train(iteration_id):
            Train the client's model on streaming data for a specified iteration.
        get_regret():
            Get the regret (loss) for each iteration.
        send_local_gradient_to_neighbor(client_list):
            Send local gradients to neighboring clients.
        receive_neighbor_gradients(client_id, model_x, topo_weight):
            Receive gradients from a neighboring client.
        update_local_parameters():
            Update local model parameters based on received gradients.

    Attributes:
        model: The machine learning model used by the client.
        b_symmetric (bool): Flag for symmetric or asymmetric communication topology.
        topology_manager: The manager for defining communication topology.
        id (int): The unique identifier of the client.
        streaming_data (list): Streaming data for training.
        optimizer: The optimizer for training the model.
        criterion: The loss criterion used for training.
        learning_rate (float): The learning rate for gradient descent.
        iteration_number (int): The total number of iterations.
        latency (float): The communication latency.
        batch_size (int): The batch size for training.
        loss_in_each_iteration (list): List to store loss for each iteration.
        model_x: The model cache for temporary values.
        neighbors_weight_dict (dict): Dictionary to store neighboring client weights.
        neighbors_topo_weight_dict (dict): Dictionary to store neighboring client topology weights.
    """
    def __init__(
        self,
        model,
        model_cache,
        client_id,
        streaming_data,
        topology_manager,
        iteration_number,
        learning_rate,
        batch_size,
        weight_decay,
        latency,
        b_symmetric,
    ):
        """
        Initialize the ClientDSGD object.

        Args:
            model: The machine learning model used by the client.
            model_cache: The model cache used for temporary values.
            client_id (int): The unique identifier of the client.
            streaming_data (list): Streaming data for training.
            topology_manager: The manager for defining communication topology.
            iteration_number (int): The total number of iterations.
            learning_rate (float): The learning rate for gradient descent.
            batch_size (int): The batch size for training.
            weight_decay (float): The weight decay for regularization.
            latency (float): The communication latency.
            b_symmetric (bool): Flag for symmetric or asymmetric communication topology.
        """
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

        self.optimizer = torch.optim.SGD(
            self.model.parameters(), lr=learning_rate, weight_decay=weight_decay
        )
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
        """
        Train the client's model on local data for a specified iteration.

        Args:
            iteration_id (int): The current iteration.
        """
        self.optimizer.zero_grad()
        train_x = torch.from_numpy(self.streaming_data[iteration_id]["x"])
        train_y = torch.FloatTensor([self.streaming_data[iteration_id]["y"]])
        outputs = self.model(train_x)
        loss = self.criterion(outputs, train_y)  # pylint: disable=E1102
        loss.backward()
        self.optimizer.step()
        self.loss_in_each_iteration.append(loss)

    def train(self, iteration_id):
        """
        Train the client's model on streaming data for a specified iteration.

        Args:
            iteration_id (int): The current iteration.
        """
        self.optimizer.zero_grad()

        if iteration_id >= self.iteration_number:
            iteration_id = iteration_id % self.iteration_number

        train_x = torch.from_numpy(self.streaming_data[iteration_id]["x"]).float()
        # print(train_x)
        train_y = torch.FloatTensor([self.streaming_data[iteration_id]["y"]])
        outputs = self.model(train_x)
        # print(train_y)
        loss = self.criterion(outputs, train_y)  # pylint: disable=E1102
        grads_z = torch.autograd.grad(loss, self.model.parameters())

        for x_paras, g_z in zip(list(self.model_x.parameters()), grads_z):
            temp = g_z.data.mul(0 - self.learning_rate)
            x_paras.data.add_(temp)

        self.loss_in_each_iteration.append(loss)

    def get_regret(self):
        """
        Get the regret (loss) for each iteration.

        Returns:
            list: A list containing the loss for each iteration.
        """
        return self.loss_in_each_iteration

    # simulation
    def send_local_gradient_to_neighbor(self, client_list):
        """
        Send local gradients to neighboring clients for simulation.

        Args:
            client_list (list): List of client objects representing neighbors.
        """
        for index in range(len(self.topology)):
            if self.topology[index] != 0 and index != self.id:
                client = client_list[index]
                client.receive_neighbor_gradients(
                    self.id, self.model_x, self.topology[index]
                )

    def receive_neighbor_gradients(self, client_id, model_x, topo_weight):
        """
        Receive gradients from a neighboring client for simulation.

        Args:
            client_id (int): The identifier of the neighboring client.
            model_x: Model parameters from the neighboring client.
            topo_weight (float): Topology weight associated with the neighboring client.
        """
        self.neighbors_weight_dict[client_id] = model_x
        self.neighbors_topo_weight_dict[client_id] = topo_weight

    def update_local_parameters(self):
        """
        Update local model parameters based on received gradients.
        """
        # update x_{t+1/2}
        for x_paras in self.model_x.parameters():
            x_paras.data.mul_(self.topology[self.id])

        for client_id in self.neighbors_weight_dict.keys():
            model_x = self.neighbors_weight_dict[client_id]
            topo_weight = self.neighbors_topo_weight_dict[client_id]
            for x_paras, x_neighbor in zip(
                list(self.model_x.parameters()), list(model_x.parameters())
            ):
                temp = x_neighbor.data.mul(topo_weight)
                x_paras.data.add_(temp)

        # update parameter z (self.model)
        for x_params, z_params in zip(
            list(self.model_x.parameters()), list(self.model.parameters())
        ):
            z_params.data.copy_(x_params)
