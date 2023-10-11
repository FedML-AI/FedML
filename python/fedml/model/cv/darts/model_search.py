import torch
import torch.nn as nn
import torch.nn.functional as F

from .genotypes import PRIMITIVES, Genotype
from .operations import OPS, FactorizedReduce, ReLUConvBN
from .utils import count_parameters_in_MB


import torch.nn as nn

class MixedOp(nn.Module):
    """
    Mixed Operation Module for Neural Architecture Search (NAS).

    This module represents a mixture of different operations and allows for dynamic selection of one
    of these operations based on a set of weights.

    Args:
        C (int): Number of input channels.
        stride (int): The stride for the operations.

    Input:
        - Input tensor `x` of shape (batch_size, C, H, W), where `C` is the number of input channels,
          and `H` and `W` are the spatial dimensions.

    Output:
        - Output tensor of shape (batch_size, C, H', W'), where `C` is the number of output channels,
          and `H'` and `W'` are the spatial dimensions after applying the selected operation.

    Attributes:
        - _ops (nn.ModuleList): A list of operations to be mixed based on weights.

    Note:
        - This module is typically used in Neural Architecture Search (NAS) to create a mixed operation
          that combines different operations (e.g., convolution, pooling) and allows the architecture
          search algorithm to learn which operations to use.

    Example:
        To create an instance of the MixedOp module and use it in a NAS cell:
        >>> mixed_op = MixedOp(C=64, stride=1)
        >>> input_tensor = torch.randn(1, 64, 32, 32)  # Example input tensor
        >>> weights = torch.randn(5)  # Example operation mixing weights
        >>> output = mixed_op(input_tensor, weights)  # Apply the mixed operation to the input

    """

    def __init__(self, C, stride):
        super(MixedOp, self).__init__()
        self._ops = nn.ModuleList()
        for primitive in PRIMITIVES:
            op = OPS[primitive](C, stride, False)
            if "pool" in primitive:
                op = nn.Sequential(op, nn.BatchNorm2d(C, affine=False))
            self._ops.append(op)

    def forward(self, x, weights):
        """
        Forward pass of the MixedOp module.

        Args:
            x (Tensor): Input tensor of shape (batch_size, C, H, W).
            weights (Tensor): Operation mixing weights of shape (num_operations,).

        Returns:
            output (Tensor): Output tensor of shape (batch_size, C, H', W').

        """
        # Apply the selected operation based on the given weights
        return sum(w * op(x) for w, op in zip(weights, self._ops))


class Cell(nn.Module):
    """
    Cell Module for Neural Architecture Search (NAS).

    This module represents a cell in a neural network architecture designed for NAS. It contains a sequence
    of mixed operations and is used to create the architecture search space.

    Args:
        steps (int): The number of steps (operations) in the cell.
        multiplier (int): The multiplier for the number of output channels.
        C_prev_prev (int): Number of input channels from two steps back.
        C_prev (int): Number of input channels from the previous step.
        C (int): Number of output channels.
        reduction (bool): Whether the cell performs reduction (downsampling).
        reduction_prev (bool): Whether the previous cell performs reduction.

    Input:
        - Two input tensors `s0` and `s1` of shape (batch_size, C_prev_prev, H, W) and (batch_size, C_prev, H, W),
          where `C_prev_prev` is the number of input channels from two steps back, `C_prev` is the number of input
          channels from the previous step, and `H` and `W` are the spatial dimensions.

    Output:
        - Output tensor of shape (batch_size, C, H', W'), where `C` is the number of output channels,
          and `H'` and `W'` are the spatial dimensions after applying the cell operations.

    Attributes:
        - preprocess0 (nn.Module): Preprocessing layer for input `s0`.
        - preprocess1 (nn.Module): Preprocessing layer for input `s1`.
        - _steps (int): The number of steps (operations) in the cell.
        - _multiplier (int): The multiplier for the number of output channels.
        - _ops (nn.ModuleList): List of mixed operations to be applied in the cell.

    Note:
        - This module is typically used in Neural Architecture Search (NAS) to create cells with different
          combinations of operations, which are then combined to form a complete neural network architecture.

    Example:
        To create an instance of the Cell module and use it in an NAS network:
        >>> cell = Cell(steps=4, multiplier=4, C_prev_prev=48, C_prev=48, C=192, reduction=False, reduction_prev=True)
        >>> input_s0 = torch.randn(1, 48, 32, 32)  # Example input tensor s0
        >>> input_s1 = torch.randn(1, 48, 32, 32)  # Example input tensor s1
        >>> weights = torch.randn(14)  # Example operation mixing weights
        >>> output = cell(input_s0, input_s1, weights)  # Apply the cell operations to the inputs

    """

    def __init__(
        self, steps, multiplier, C_prev_prev, C_prev, C, reduction, reduction_prev
    ):
        super(Cell, self).__init__()
        self.reduction = reduction

        if reduction_prev:
            self.preprocess0 = FactorizedReduce(C_prev_prev, C, affine=False)
        else:
            self.preprocess0 = ReLUConvBN(C_prev_prev, C, 1, 1, 0, affine=False)
        self.preprocess1 = ReLUConvBN(C_prev, C, 1, 1, 0, affine=False)
        self._steps = steps
        self._multiplier = multiplier

        self._ops = nn.ModuleList()
        
        for i in range(self._steps):
            for j in range(2 + i):
                stride = 2 if reduction and j < 2 else 1
                op = MixedOp(C, stride)
                self._ops.append(op)

    def forward(self, s0, s1, weights):
        """
        Forward pass of the Cell module.

        Args:
            s0 (Tensor): Input tensor s0 of shape (batch_size, C_prev_prev, H, W).
            s1 (Tensor): Input tensor s1 of shape (batch_size, C_prev, H, W).
            weights (Tensor): Operation mixing weights of shape (num_operations,).

        Returns:
            output (Tensor): Output tensor of shape (batch_size, C, H', W').

        """
        s0 = self.preprocess0(s0)
        s1 = self.preprocess1(s1)

        states = [s0, s1]
        offset = 0
        for i in range(self._steps):
            s = sum(
                self._ops[offset + j](h, weights[offset + j])
                for j, h in enumerate(states)
            )
            offset += len(states)
            states.append(s)

        return torch.cat(states[-self._multiplier :], dim=1)


class InnerCell(nn.Module):
    """
    InnerCell Module for Neural Architecture Search (NAS).

    This module represents an inner cell in a neural network architecture designed for NAS. It contains a sequence
    of mixed operations and is used to create the architecture search space.

    Args:
        steps (int): The number of steps (operations) in the inner cell.
        multiplier (int): The multiplier for the number of output channels.
        C_prev_prev (int): Number of input channels from two steps back.
        C_prev (int): Number of input channels from the previous step.
        C (int): Number of output channels.
        reduction (bool): Whether the inner cell performs reduction (downsampling).
        reduction_prev (bool): Whether the previous cell performs reduction.
        weights (Tensor): Operation mixing weights for the inner cell.

    Input:
        - Two input tensors `s0` and `s1` of shape (batch_size, C_prev_prev, H, W) and (batch_size, C_prev, H, W),
          where `C_prev_prev` is the number of input channels from two steps back, `C_prev` is the number of input
          channels from the previous step, and `H` and `W` are the spatial dimensions.

    Output:
        - Output tensor of shape (batch_size, C, H', W'), where `C` is the number of output channels,
          and `H'` and `W'` are the spatial dimensions after applying the inner cell operations.

    Attributes:
        - preprocess0 (nn.Module): Preprocessing layer for input `s0`.
        - preprocess1 (nn.Module): Preprocessing layer for input `s1`.
        - _steps (int): The number of steps (operations) in the inner cell.
        - _multiplier (int): The multiplier for the number of output channels.
        - _ops (nn.ModuleList): List of mixed operations to be applied in the inner cell.

    Note:
        - This module is typically used in Neural Architecture Search (NAS) to create inner cells with different
          combinations of operations, which are then combined to form a complete neural network architecture.

    Example:
        To create an instance of the InnerCell module and use it in an NAS network:
        >>> inner_cell = InnerCell(steps=4, multiplier=4, C_prev_prev=48, C_prev=48, C=192, reduction=False,
        ...                        reduction_prev=True, weights=weights)
        >>> input_s0 = torch.randn(1, 48, 32, 32)  # Example input tensor s0
        >>> input_s1 = torch.randn(1, 48, 32, 32)  # Example input tensor s1
        >>> output = inner_cell(input_s0, input_s1)  # Apply the inner cell operations to the inputs

    """

    def __init__(
        self,
        steps,
        multiplier,
        C_prev_prev,
        C_prev,
        C,
        reduction,
        reduction_prev,
        weights,
    ):
        super(InnerCell, self).__init__()
        self.reduction = reduction

        if reduction_prev:
            self.preprocess0 = FactorizedReduce(C_prev_prev, C, affine=False)
        else:
            self.preprocess0 = ReLUConvBN(C_prev_prev, C, 1, 1, 0, affine=False)
        self.preprocess1 = ReLUConvBN(C_prev, C, 1, 1, 0, affine=False)
        self._steps = steps
        self._multiplier = multiplier

        self._ops = nn.ModuleList()

        offset = 0
        keys = list(OPS.keys())
        for i in range(self._steps):
            for j in range(2 + i):
                stride = 2 if reduction and j < 2 else 1
                weight = weights.data[offset + j]
                choice = keys[weight.argmax()]
                op = OPS[choice](C, stride, False)
                if "pool" in choice:
                    op = nn.Sequential(op, nn.BatchNorm2d(C, affine=False))
                self._ops.append(op)
            offset += i + 2

    def forward(self, s0, s1):
        """
        Forward pass of the InnerCell module.

        Args:
            s0 (Tensor): Input tensor s0 of shape (batch_size, C_prev_prev, H, W).
            s1 (Tensor): Input tensor s1 of shape (batch_size, C_prev, H, W).

        Returns:
            output (Tensor): Output tensor of shape (batch_size, C, H', W').

        """
        s0 = self.preprocess0(s0)
        s1 = self.preprocess1(s1)

        states = [s0, s1]
        offset = 0
        for i in range(self._steps):
            s = sum(self._ops[offset + j](h) for j, h in enumerate(states))
            offset += len(states)
            states.append(s)

        return torch.cat(states[-self._multiplier :], dim=1)


class ModelForModelSizeMeasure(nn.Module):
    """
    Model used solely for measuring the size of the generated model.

    This class is designed to calculate the size of a model based on specific choices of operations determined by
    the alpha values of the DARTS model. It serves the purpose of estimating the model size without performing
    actual training or inference.

    Differences from the DARTS model:
    1. Additional parameters "alphas_normal" and "alphas_reduce" are required in the constructor.
    2. The Cell module combines the functionality of both Cell and MixedOp.
    3. MixedOp is replaced with a fixed choice of operation based on the argmax(alpha_values).
    4. The Cell class is redefined as an inner class with the same name.

    Args:
        C (int): The number of channels in the input data.
        num_classes (int): The number of output classes.
        layers (int): The number of layers in the model.
        criterion: The loss criterion used for training.
        alphas_normal (Tensor): Alpha values for normal cells.
        alphas_reduce (Tensor): Alpha values for reduction cells.
        steps (int, optional): The number of steps (operations) in each cell. Default is 4.
        multiplier (int, optional): The multiplier for the number of output channels. Default is 4.
        stem_multiplier (int, optional): The multiplier for the number of channels in the stem. Default is 3.

    Input:
        - Input tensor of shape (batch_size, 3, H, W), where `batch_size` is the number of input samples,
          `H` and `W` are the spatial dimensions, and `3` represents the RGB channels.

    Output:
        - Output tensor of shape (batch_size, num_classes), representing class predictions.

    Attributes:
        - stem (nn.Sequential): Stem layer consisting of a convolutional layer and batch normalization.
        - cells (nn.ModuleList): List of inner cells that make up the model.
        - global_pooling (nn.AdaptiveAvgPool2d): Global pooling layer for spatial aggregation.
        - classifier (nn.Linear): Fully connected layer for class prediction.

    Note:
        - This class is primarily used for measuring the size of a model and does not perform training or inference.

    Example:
        To create an instance of the ModelForModelSizeMeasure and use it to measure the model size:
        >>> model = ModelForModelSizeMeasure(C=16, num_classes=10, layers=8, criterion=nn.CrossEntropyLoss(),
        ...                                  alphas_normal=alphas_normal, alphas_reduce=alphas_reduce)
        >>> input_data = torch.randn(1, 3, 32, 32)  # Example input tensor
        >>> model_size = get_model_size(model, input_data)  # Get the estimated model size

    """

    def __init__(
        self,
        C,
        num_classes,
        layers,
        criterion,
        alphas_normal,
        alphas_reduce,
        steps=4,
        multiplier=4,
        stem_multiplier=3,
    ):
        super(ModelForModelSizeMeasure, self).__init__()
        self._C = C
        self._num_classes = num_classes
        self._layers = layers
        self._criterion = criterion
        self._steps = steps
        self._multiplier = multiplier

        C_curr = stem_multiplier * C  # 3*16
        self.stem = nn.Sequential(
            nn.Conv2d(3, C_curr, 3, padding=1, bias=False), nn.BatchNorm2d(C_curr)
        )

        C_prev_prev, C_prev, C_curr = C_curr, C_curr, C
        self.cells = nn.ModuleList()
        reduction_prev = False


        for i in range(layers):
            if i in [layers // 3, 2 * layers // 3]:
                C_curr *= 2
                reduction = True
                cell = InnerCell(
                    steps,
                    multiplier,
                    C_prev_prev,
                    C_prev,
                    C_curr,
                    reduction,
                    reduction_prev,
                    alphas_reduce,
                )
            else:
                reduction = False
                cell = InnerCell(
                    steps,
                    multiplier,
                    C_prev_prev,
                    C_prev,
                    C_curr,
                    reduction,
                    reduction_prev,
                    alphas_normal,
                )

            reduction_prev = reduction
            self.cells += [cell]
            C_prev_prev, C_prev = C_prev, multiplier * C_curr

        self.global_pooling = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(C_prev, num_classes)

    def forward(self, input_data):
        s0 = s1 = self.stem(input_data)
        for i, cell in enumerate(self.cells):
            if cell.reduction:
                s0, s1 = s1, cell(s0, s1)
            else:
                s0, s1 = s1, cell(s0, s1)
        out = self.global_pooling(s1)
        logits = self.classifier(out.view(out.size(0), -1))
        return logits


class Network(nn.Module):
    """
    DARTS-based neural network model for image classification.

    Args:
        C (int): The number of channels in the input data.
        num_classes (int): The number of output classes.
        layers (int): The number of layers in the model.
        criterion: The loss criterion used for training.
        steps (int, optional): The number of steps (operations) in each cell. Default is 4.
        multiplier (int, optional): The multiplier for the number of output channels. Default is 4.
        stem_multiplier (int, optional): The multiplier for the number of channels in the stem. Default is 3.

    Input:
        - Input tensor of shape (batch_size, 3, H, W), where `batch_size` is the number of input samples,
          `H` and `W` are the spatial dimensions, and `3` represents the RGB channels.

    Output:
        - Output tensor of shape (batch_size, num_classes), representing class predictions.

    Attributes:
        - stem (nn.Sequential): Stem layer consisting of a convolutional layer and batch normalization.
        - cells (nn.ModuleList): List of inner cells that make up the model.
        - global_pooling (nn.AdaptiveAvgPool2d): Global pooling layer for spatial aggregation.
        - classifier (nn.Linear): Fully connected layer for class prediction.
        - alphas_normal (nn.Parameter): Learnable alpha values for normal cells.
        - alphas_reduce (nn.Parameter): Learnable alpha values for reduction cells.

    Methods:
        - new(self): Create a new instance of the network with the same architecture and initialize alpha values.
        - new_arch_parameters(self): Generate new architecture parameters (alphas) for the network.
        - arch_parameters(self): Get the current architecture parameters (alphas) of the network.
        - genotype(self): Get the genotype of the network, which describes the architecture.
        - get_current_model_size(self): Estimate the current model size in megabytes.

    Note:
        - This class is based on the DARTS (Differentiable Architecture Search) architecture and is used for
          neural architecture search (NAS) experiments.

    Example:
        To create an instance of the Network class and use it for architecture search:
        >>> model = Network(C=16, num_classes=10, layers=8, criterion=nn.CrossEntropyLoss())
        >>> input_data = torch.randn(1, 3, 32, 32)  # Example input tensor
        >>> genotype, normal_count, reduce_count = model.genotype()  # Get the architecture genotype
        >>> model_size = model.get_current_model_size()  # Get the estimated model size
    """

    def __init__(
        self,
        C,
        num_classes,
        layers,
        criterion,
        steps=4,
        multiplier=4,
        stem_multiplier=3,
    ):
        super(Network, self).__init__()
        print(Network)
        self._C = C
        self._num_classes = num_classes
        self._layers = layers
        self._criterion = criterion
        self._steps = steps
        self._multiplier = multiplier
        self._stem_multiplier = stem_multiplier


        C_curr = stem_multiplier * C  # 3*16
        self.stem = nn.Sequential(
            nn.Conv2d(3, C_curr, 3, padding=1, bias=False), nn.BatchNorm2d(C_curr)
        )

        C_prev_prev, C_prev, C_curr = C_curr, C_curr, C
        self.cells = nn.ModuleList()
        reduction_prev = False

        # for layers = 8, when layer_i = 2, 5, the cell is reduction cell.
        for i in range(layers):
            if i in [layers // 3, 2 * layers // 3]:
                C_curr *= 2
                reduction = True
            else:
                reduction = False
            cell = Cell(
                steps,
                multiplier,
                C_prev_prev,
                C_prev,
                C_curr,
                reduction,
                reduction_prev,
            )
            reduction_prev = reduction
            self.cells += [cell]
            C_prev_prev, C_prev = C_prev, multiplier * C_curr

        self.global_pooling = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(C_prev, num_classes)

        self._initialize_alphas()

    def new(self):
        """
        Create a new instance of the network with the same architecture and initialize alpha values.

        Returns:
            Network: A new instance of the Network class with the same architecture.
        """
        model_new = Network(
            self._C, self._num_classes, self._layers, self._criterion, self.device
        ).to(self.device)
        for x, y in zip(model_new.arch_parameters(), self.arch_parameters()):
            x.data.copy_(y.data)
        return model_new

    def forward(self, input):
        """
        Forward pass of the neural network.

        Args:
            input (Tensor): Input tensor of shape (batch_size, 3, H, W), where `batch_size` is the number of
                            input samples, `H` and `W` are the spatial dimensions, and `3` represents the RGB channels.

        Returns:
            Tensor: Output tensor of shape (batch_size, num_classes), representing class predictions.
        """
        s0 = s1 = self.stem(input)
        for i, cell in enumerate(self.cells):
            if cell.reduction:
                weights = F.softmax(self.alphas_reduce, dim=-1)
            else:
                weights = F.softmax(self.alphas_normal, dim=-1)
            s0, s1 = s1, cell(s0, s1, weights)
        out = self.global_pooling(s1)
        logits = self.classifier(out.view(out.size(0), -1))
        return logits

    def _initialize_alphas(self):
        """
        Initialize alpha values for normal and reduction cells.
        """
        k = sum(1 for i in range(self._steps) for n in range(2 + i))
        num_ops = len(PRIMITIVES)

        self.alphas_normal = nn.Parameter(1e-3 * torch.randn(k, num_ops))
        self.alphas_reduce = nn.Parameter(1e-3 * torch.randn(k, num_ops))
        self._arch_parameters = [
            self.alphas_normal,
            self.alphas_reduce,
        ]

    def new_arch_parameters(self):
        """
        Generate new architecture parameters (alphas) for the network.

        Returns:
            List[nn.Parameter]: List of architecture parameters (alphas).
        """
        k = sum(1 for i in range(self._steps) for n in range(2 + i))
        num_ops = len(PRIMITIVES)

        alphas_normal = nn.Parameter(1e-3 * torch.randn(k, num_ops)).to(self.device)
        alphas_reduce = nn.Parameter(1e-3 * torch.randn(k, num_ops)).to(self.device)
        _arch_parameters = [
            alphas_normal,
            alphas_reduce,
        ]
        return _arch_parameters

    def arch_parameters(self):
        """
        Get the current architecture parameters (alphas) of the network.

        Returns:
            List[nn.Parameter]: List of architecture parameters (alphas).
        """
        return self._arch_parameters

    def genotype(self):
        """
        Get the genotype of the network, which describes the architecture.

        Returns:
            Genotype: The genotype of the network.
        """
        def _isCNNStructure(k_best):
            return k_best >= 4

        def _parse(weights):
            gene = []
            n = 2
            start = 0
            cnn_structure_count = 0
            for i in range(self._steps):
                end = start + n
                W = weights[start:end].copy()
                edges = sorted(
                    range(i + 2),
                    key=lambda x: -max(
                        W[x][k]
                        for k in range(len(W[x]))
                        if k != PRIMITIVES.index("none")
                    ),
                )[:2]
                for j in edges:
                    k_best = None
                    for k in range(len(W[j])):
                        if k != PRIMITIVES.index("none"):
                            if k_best is None or W[j][k] > W[j][k_best]:
                                k_best = k

                    if _isCNNStructure(k_best):
                        cnn_structure_count += 1
                    gene.append((PRIMITIVES[k_best], j))
                start = end
                n += 1
            return gene, cnn_structure_count

        with torch.no_grad():
            gene_normal, cnn_structure_count_normal = _parse(
                F.softmax(self.alphas_normal, dim=-1).data.cpu().numpy()
            )
            gene_reduce, cnn_structure_count_reduce = _parse(
                F.softmax(self.alphas_reduce, dim=-1).data.cpu().numpy()
            )

            concat = range(2 + self._steps - self._multiplier, self._steps + 2)
            genotype = Genotype(
                normal=gene_normal,
                normal_concat=concat,
                reduce=gene_reduce,
                reduce_concat=concat,
            )
        return genotype, cnn_structure_count_normal, cnn_structure_count_reduce

    def get_current_model_size(self):
        """
        Estimate the current model size in megabytes.

        Returns:
            float: The estimated model size in megabytes.
        """
        model = ModelForModelSizeMeasure(
            self._C,
            self._num_classes,
            self._layers,
            self._criterion,
            self.alphas_normal,
            self.alphas_reduce,
            self._steps,
            self._multiplier,
            self._stem_multiplier,
        )
        size = count_parameters_in_MB(model)
        # This need to be further checked with cuda stuff
        del model
        return size
