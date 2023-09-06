import random

import numpy as np
import torch


class ClientPushsum(object):
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
        time_varying,
    ):
        """
        Initialize a ClientPushsum instance.

        Args:
            model: The client's model.
            model_cache: Cache for the model parameters.
            client_id (int): Identifier for the client.
            streaming_data: Streaming data for training.
            topology_manager: Topology manager for network topology.
            iteration_number (int): Number of iterations.
            learning_rate (float): Learning rate for optimization.
            batch_size (int): Batch size for training.
            weight_decay (float): Weight decay for optimization.
            latency (float): Latency in communication.
            b_symmetric (bool): Whether the topology is symmetric.
            time_varying (bool): Whether the topology is time-varying.
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
        self.time_varying = time_varying

        self.optimizer = torch.optim.SGD(
            self.model.parameters(), lr=learning_rate, weight_decay=weight_decay
        )
        self.criterion = torch.nn.BCELoss()

        self.learning_rate = learning_rate

        self.iteration_number = iteration_number

        self.batch_size = batch_size
        self.loss_in_each_iteration = []

        self.omega = 1

        # the default weight of the model is z_t, while the x weight is another weight used as temporary value
        self.model_x = model_cache

        # neighbors_weight_dict
        self.neighbors_weight_dict = dict()
        self.neighbors_omega_dict = dict()
        self.neighbors_topo_weight_dict = dict()

    def train_local(self, iteration_id):
        """
        Train the client's model using local data for a specific iteration.

        Args:
            iteration_id (int): The iteration index.
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
        Train the client's model using data for a specific iteration.

        Args:
            iteration_id (int): The iteration index.
        """
        self.optimizer.zero_grad()

        if iteration_id >= self.iteration_number:
            iteration_id = iteration_id % self.iteration_number

        # update the confusion matrix
        if self.time_varying:
            seed = iteration_id
            random.seed(seed)
            np.random.seed(seed)
            self.topology_manager.generate_topology()
            if self.b_symmetric:
                self.topology = self.topology_manager.get_symmetric_neighbor_list(
                    self.id
                )
            else:
                self.topology = self.topology_manager.get_asymmetric_neighbor_list(
                    self.id
                )

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
                    self.id,
                    self.model_x,
                    self.topology[index],
                    self.omega * self.topology[index],
                )

    def receive_neighbor_gradients(self, client_id, model_x, topo_weight, omega):
        """
        Receive gradients from a neighboring client for simulation.

        Args:
            client_id (int): The identifier of the neighboring client.
            model_x: Model parameters from the neighboring client.
            topo_weight (float): Topology weight associated with the neighboring client.
            omega (float): Omega value for push-sum.
        """
        self.neighbors_weight_dict[client_id] = model_x
        self.neighbors_topo_weight_dict[client_id] = topo_weight
        self.neighbors_omega_dict[client_id] = omega

    def update_local_parameters(self):
        """
        Update local model parameters and omega based on received gradients.
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
                # print("topo_weight=" + str(topo_weight))
                # print("x_neighbor=" + str(temp))
                x_paras.data.add_(temp)

        # update omega
        self.omega *= self.topology[self.id]
        for client_id in self.neighbors_omega_dict.keys():
            self.omega += self.neighbors_omega_dict[client_id]

        # print(self.omega)

        # update parameter z (self.model)
        for x_params, z_params in zip(
            list(self.model_x.parameters()), list(self.model.parameters())
        ):
            # print("1.0 / self.omega=" + str(1.0 / self.omega))
            temp = x_params.data.mul(1.0 / self.omega)
            z_params.data.copy_(temp)
