import copy
import logging
from typing import List, Tuple

import numpy as np
import torch


class GPUAccelerator:
    def __init__(self, model_template, validation_data, device=None, seed=0):
        if device is None:
            raise ValueError("GPUAccelerator requires an explicit device from fedml.device.get_device(args)")
        self.device = device
        self.seed = int(seed)
        self.model_template = copy.deepcopy(model_template).to(self.device)
        self.val_images, self.val_labels = validation_data
        self.val_images = self.val_images.to(self.device)
        self.val_labels = self.val_labels.to(self.device)
        self.state_keys = list(self.model_template.state_dict().keys())
        trainable_keys = {k for k, _ in self.model_template.named_parameters()}
        self.trainable_mask = [k in trainable_keys for k in self.state_keys]
        self.has_batchnorm = any(
            isinstance(module, torch.nn.modules.batchnorm._BatchNorm)
            for module in self.model_template.modules()
        )
        state_values = list(self.model_template.state_dict().values())
        self.param_shapes = [value.shape for value in state_values]
        self.param_sizes = [value.numel() for value in state_values]
        self.total_params = sum(self.param_sizes)
        self.client_params_matrix = None
        logging.info(
            "GPUAccelerator initialized | device=%s | has_batchnorm=%s | total_params=%d | "
            "val_images.shape=%s | val_labels.shape=%s",
            self.device,
            self.has_batchnorm,
            self.total_params,
            tuple(self.val_images.shape),
            tuple(self.val_labels.shape),
        )

    def _load_state_from_ndarrays(self, params: List[np.ndarray]):
        state_values = list(self.model_template.state_dict().values())
        with torch.no_grad():
            for target_tensor, array in zip(state_values, params):
                source = torch.as_tensor(array, device=self.device)
                if source.dtype != target_tensor.dtype:
                    source = source.to(dtype=target_tensor.dtype)
                target_tensor.copy_(source)

    def _extract_state_to_ndarrays(self) -> List[np.ndarray]:
        return [value.detach().cpu().numpy() for value in self.model_template.state_dict().values()]

    def recalibrate_batchnorm(self, params: List[np.ndarray], batch_size: int = 64, passes: int = 1):
        if not self.has_batchnorm:
            return params
        torch.manual_seed(self.seed)
        self._load_state_from_ndarrays(params)
        self.model_template.train()
        with torch.no_grad():
            total = int(self.val_images.shape[0])
            for _ in range(max(1, int(passes))):
                for start in range(0, total, int(batch_size)):
                    end = min(total, start + int(batch_size))
                    _ = self.model_template(self.val_images[start:end])
        self.model_template.eval()
        return self._extract_state_to_ndarrays()

    def set_client_parameters(self, client_parameters: List[List[np.ndarray]]):
        num_clients = len(client_parameters)
        self.client_params_matrix = torch.zeros(
            (num_clients, self.total_params), device=self.device, dtype=torch.float32
        )
        for index, params in enumerate(client_parameters):
            flat_params = np.concatenate([param.ravel() for param in params])
            self.client_params_matrix[index] = torch.tensor(flat_params, device=self.device)
        mem_mb = (
            self.client_params_matrix.element_size()
            * self.client_params_matrix.nelement()
            / (1024 ** 2)
        )
        logging.info(
            "GPUAccelerator client_params_matrix: shape=%s mem=%.1fMB device=%s",
            tuple(self.client_params_matrix.shape),
            mem_mb,
            self.device,
        )

    def calculate_fitness(self, alpha: np.ndarray) -> Tuple[float, float]:
        if self.client_params_matrix is None:
            raise RuntimeError("call set_client_parameters before calculate_fitness")
        alpha_tensor = torch.tensor(alpha, dtype=torch.float32, device=self.device)
        aggregated_flat = torch.matmul(alpha_tensor, self.client_params_matrix)
        reconstructed_params = []
        start = 0
        for shape, size in zip(self.param_shapes, self.param_sizes):
            end = start + size
            reconstructed_params.append(aggregated_flat[start:end].view(shape))
            start = end
        with torch.no_grad():
            for target_tensor, new_value in zip(self.model_template.state_dict().values(), reconstructed_params):
                target_tensor.copy_(new_value.to(dtype=target_tensor.dtype))
        self.model_template.eval()
        criterion = torch.nn.CrossEntropyLoss()
        with torch.no_grad():
            outputs = self.model_template(self.val_images)
            loss = criterion(outputs, self.val_labels)
            model_norm = torch.tensor(0.0, device=self.device)
            for param in self.model_template.parameters():
                model_norm += torch.sum(param ** 2)
            model_norm = torch.sqrt(model_norm)
        return float(loss.item()), float(model_norm.item())
