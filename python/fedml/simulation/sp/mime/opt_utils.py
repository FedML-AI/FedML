import logging

import torch


def show_opt_state(optimizer):
    """
    Display selected optimizer's state information.

    Args:
        optimizer: The optimizer to display state information for.
    """
    i = 0
    for p in optimizer.state.keys():
        # print(list(optimizer.state[p].keys()))
        if i > 5:
            break
        i +=1
        for key in optimizer.state[p].keys():
            # print(key, type(optimizer.state[p][key]))
            if isinstance(optimizer.state[p][key], int):
                print(key, (optimizer.state[p][key]))
            else:
                print(key, torch.norm((optimizer.state[p][key])))

def show_named_state(named_states):
    """
    Display state information for a dictionary of named states.

    Args:
        named_states (dict): A dictionary containing named states to display.
    """
    i = 0
    for name in named_states.keys():
        # print(list(optimizer.state[p].keys()))
        if i > 5:
            break
        i +=1
        for key in named_states[name].keys():
            # print(key, type(optimizer.state[p][key]))
            if isinstance(named_states[name][key], int):
                print(name, key, (named_states[name][key]))
            else:
                print(name, key, torch.norm((named_states[name][key])))



class OptimizerLoader():
    def __init__(self, model, optimizer):
        """
        Initialize the OptimizerLoader.

        Args:
            model: The model being optimized.
            optimizer: The optimizer used for training.
        """
        self.optimizer = optimizer
        self.model = model
        self.named_states = {}
        self.parameter_names = {}
        for name, parameter in model.named_parameters():
            self.named_states[name] = optimizer.state[parameter]
            self.parameter_names[parameter] = name

        # for p in optimizer.state.keys():
        # #     print(list(optimizer.state[p].keys()))
        #     for key in optimizer.state[p].keys():
        #         print(key, type(optimizer.state[p][key]))

    def get_opt_state(self):
        """
        Get the optimizer's named states.

        Returns:
            dict: A dictionary containing the optimizer's named states.
        """
        return self.named_states

    def set_opt_state(self, named_states, device="cpu"):
        """
        Set the optimizer's named states.

        Args:
            named_states (dict): A dictionary containing the named states to set.
            device (str): The target device for the named states (default is "cpu").
        """
        for p in self.optimizer.state.keys():
            new_state = named_states[self.parameter_names[p]]
            # for key in self.optimizer.state[p].keys():
            for key in new_state.keys():
                self.optimizer.state[p][key] = new_state[key].to(device)
                # print(key, type(self.optimizer.state[p][key]))

    def get_grad(self):
        """
        Get the gradients of the model's parameters.

        Returns:
            dict: A dictionary containing the gradients of the model's parameters.
        """
        grad = {}
        for name, parameter in self.model.named_parameters():
            grad[name] = parameter.grad
        return grad

    def set_grad(self, grad, device="cpu"):
        """
        Set the gradients of the model's parameters.

        Args:
            grad (dict): A dictionary containing the gradients to set.
            device (str): The target device for the gradients (default is "cpu").
        """
        for name, parameter in self.model.named_parameters():
            # logging.info(f"parameter.grad: {type(parameter.grad)}, grad[name]: {type(grad[name])} ")
            # logging.info(f"parameter.grad.shape: {parameter.grad.shape}, grad[name].shape: {grad[name].shape} ")
            # logging.info(f"parameter.grad: {parameter.grad.dtype}, grad[name]: {grad[name].dtype} ")
            # logging.info(f"parameter.grad.device: {parameter.grad.device}, grad[name].device: {grad[name].device} ")
            parameter.grad = grad[name].to(device)
        return grad

    def zero_grad(self):
        """
        Zero out the gradients of the model's parameters.
        """
        self.optimizer.zero_grad()

    def update_opt_state(self, update_model=False):
        """
        Update the optimizer's state after a step.

        Args:
            update_model (bool): Whether to update the model's parameters as well (default is False).

        Returns:
            dict: A dictionary containing the updated optimizer's named states.
        """
        if not update_model:
            origin_model_params = self.model.state_dict()
        self.optimizer.step()
        if not update_model:
            self.model.load_state_dict(origin_model_params)
        return self.get_opt_state()









