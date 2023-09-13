import torch
import torch.nn as nn

from .operations import FactorizedReduce, ReLUConvBN, OPS, Identity
from .utils import drop_path


class Cell(nn.Module):
    """
    Cell in a neural architecture described by a genotype.

    Args:
        genotype (Genotype): Genotype describing the cell's architecture.
        C_prev_prev (int): Number of input channels from two steps back.
        C_prev (int): Number of input channels from the previous step.
        C (int): Number of output channels.
        reduction (bool): Whether the cell is a reduction cell.
        reduction_prev (bool): Whether the previous cell was a reduction cell.

    Input:
        - s0 (Tensor): Input tensor from two steps back, shape (batch_size, C_prev_prev, H, W).
        - s1 (Tensor): Input tensor from the previous step, shape (batch_size, C_prev, H, W).
        - drop_prob (float): Dropout probability for drop path regularization during training.

    Output:
        - Output tensor of the cell, shape (batch_size, C, H, W).

    """

    def __init__(self, genotype, C_prev_prev, C_prev, C, reduction, reduction_prev):
        super(Cell, self).__init__()

        if reduction_prev:
            self.preprocess0 = FactorizedReduce(C_prev_prev, C)
        else:
            self.preprocess0 = ReLUConvBN(C_prev_prev, C, 1, 1, 0)
        self.preprocess1 = ReLUConvBN(C_prev, C, 1, 1, 0)

        if reduction:
            op_names, indices = zip(*genotype.reduce)
            concat = genotype.reduce_concat
        else:
            op_names, indices = zip(*genotype.normal)
            concat = genotype.normal_concat
        self._compile(C, op_names, indices, concat, reduction)

    def _compile(self, C, op_names, indices, concat, reduction):
        """
        Compiles the operations for the cell based on the given genotype.

        Args:
            C (int): Number of output channels for the cell.
            op_names (list of str): Names of the operations for each edge in the cell.
            indices (list of int): Indices of the operations for each edge in the cell.
            concat (list of int): Concatenation points for the cell.
            reduction (bool): Whether the cell is a reduction cell.

        """
        assert len(op_names) == len(indices)
        self._steps = len(op_names) // 2
        self._concat = concat
        self.multiplier = len(concat)

        self._ops = nn.ModuleList()
        for name, index in zip(op_names, indices):
            stride = 2 if reduction and index < 2 else 1
            op = OPS[name](C, stride, True)
            self._ops += [op]
        self._indices = indices

    def forward(self, s0, s1, drop_prob):
        """
        Forward pass through the cell.

        Args:
            s0 (Tensor): Input tensor from two steps back, shape (batch_size, C_prev_prev, H, W).
            s1 (Tensor): Input tensor from the previous step, shape (batch_size, C_prev, H, W).
            drop_prob (float): Dropout probability for drop path regularization during training.

        Returns:
            Tensor: Output tensor of the cell, shape (batch_size, C, H, W).

        """
        s0 = self.preprocess0(s0)
        s1 = self.preprocess1(s1)

        states = [s0, s1]
        for i in range(self._steps):
            h1 = states[self._indices[2 * i]]
            h2 = states[self._indices[2 * i + 1]]
            op1 = self._ops[2 * i]
            op2 = self._ops[2 * i + 1]
            h1 = op1(h1)
            h2 = op2(h2)
            if self.training and drop_prob > 0.0:
                if not isinstance(op1, Identity):
                    h1 = drop_path(h1, drop_prob)
                if not isinstance(op2, Identity):
                    h2 = drop_path(h2, drop_prob)
            s = h1 + h2
            states += [s]
        return torch.cat([states[i] for i in self._concat], dim=1)



class AuxiliaryHeadCIFAR(nn.Module):
    """
    Auxiliary head for CIFAR classification in the DARTS model.

    Args:
        C (int): Number of input channels.
        num_classes (int): Number of classes for classification.

    Input:
        - Input tensor of shape (batch_size, C, 8, 8), assuming an input size of 8x8.

    Output:
        - Output tensor of shape (batch_size, num_classes), representing class scores.

    Architecture:
        - ReLU activation
        - Average pooling with 5x5 kernel and stride 3 (resulting in an image size of 2x2)
        - 1x1 convolution with 128 output channels
        - Batch normalization
        - ReLU activation
        - 2x2 convolution with 768 output channels
        - Batch normalization
        - ReLU activation
        - Linear layer with num_classes output units for classification.

    """

    def __init__(self, C, num_classes):
        
        super(AuxiliaryHeadCIFAR, self).__init__()
        self.features = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.AvgPool2d(5, stride=3, padding=0, count_include_pad=False),
            nn.Conv2d(C, 128, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 768, 2, bias=False),
            nn.BatchNorm2d(768),
            nn.ReLU(inplace=True),
        )
        self.classifier = nn.Linear(768, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x.view(x.size(0), -1))
        return x


class AuxiliaryHeadImageNet(nn.Module):
    def __init__(self, C, num_classes):
        """assuming input size 14x14"""
        super(AuxiliaryHeadImageNet, self).__init__()
        self.features = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.AvgPool2d(5, stride=2, padding=0, count_include_pad=False),
            nn.Conv2d(C, 128, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 768, 2, bias=False),
            # NOTE: This batchnorm was omitted in my earlier implementation due to a typo.
            # Commenting it out for consistency with the experiments in the paper.
            # nn.BatchNorm2d(768),
            nn.ReLU(inplace=True),
        )
        self.classifier = nn.Linear(768, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x.view(x.size(0), -1))
        return x


class NetworkCIFAR(nn.Module):
    """
    DARTS network architecture for CIFAR dataset.

    Args:
        C (int): Initial number of channels.
        num_classes (int): Number of classes for classification.
        layers (int): Number of layers.
        auxiliary (bool): Whether to use auxiliary heads.
        genotype (Genotype): Genotype specifying the cell structure.

    Input:
        - Input tensor of shape (batch_size, 3, 32, 32), where 3 is for RGB channels.

    Output:
        - Main network output tensor of shape (batch_size, num_classes).
        - Auxiliary head output tensor if auxiliary is True and during training.

    Architecture:
        - Stem: Initial convolution layer followed by batch normalization.
        - Cells: Stack of cells with specified genotype.
        - Auxiliary Head: Optional auxiliary head for training stability.
        - Global Pooling: Adaptive average pooling to 1x1 size.
        - Classifier: Linear layer for classification.

    """

    def __init__(self, C, num_classes, layers, auxiliary, genotype):
        super(NetworkCIFAR, self).__init__()
        self._layers = layers
        self._auxiliary = auxiliary

        self.drop_path_prob = 0.5

        stem_multiplier = 3
        C_curr = stem_multiplier * C
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
            else:
                reduction = False
            cell = Cell(
                genotype, C_prev_prev, C_prev, C_curr, reduction, reduction_prev
            )
            reduction_prev = reduction
            self.cells += [cell]
            C_prev_prev, C_prev = C_prev, cell.multiplier * C_curr
            if i == 2 * layers // 3:
                C_to_auxiliary = C_prev

        if auxiliary:
            self.auxiliary_head = AuxiliaryHeadCIFAR(C_to_auxiliary, num_classes)
        self.global_pooling = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(C_prev, num_classes)

    def forward(self, input):
        logits_aux = None
        s0 = s1 = self.stem(input)
        for i, cell in enumerate(self.cells):
            s0, s1 = s1, cell(s0, s1, self.drop_path_prob)
            if i == 2 * self._layers // 3:
                if self._auxiliary and self.training:
                    logits_aux = self.auxiliary_head(s1)
        out = self.global_pooling(s1)
        logits = self.classifier(out.view(out.size(0), -1))
        return logits, logits_aux


class NetworkImageNet(nn.Module):
    """
    Network architecture for ImageNet dataset.

    Args:
        C (int): Initial number of channels.
        num_classes (int): Number of classes for classification.
        layers (int): Number of layers.
        auxiliary (bool): Whether to include an auxiliary head.
        genotype (Genotype): Genotype specifying the cell structure.

    Input:
        - Input tensor of shape (batch_size, 3, height, width).

    Output:
        - Main classifier logits tensor of shape (batch_size, num_classes).
        - Auxiliary classifier logits tensor if auxiliary is True, otherwise None.

    """

    def __init__(self, C, num_classes, layers, auxiliary, genotype):
        super(NetworkImageNet, self).__init__()
        self._layers = layers
        self._auxiliary = auxiliary
        self.drop_path_prob = 0.5

        self.stem0 = nn.Sequential(
            nn.Conv2d(3, C // 2, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(C // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(C // 2, C, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(C),
        )

        self.stem1 = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Conv2d(C, C, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(C),
        )

        C_prev_prev, C_prev, C_curr = C, C, C

        self.cells = nn.ModuleList()
        reduction_prev = True
        for i in range(layers):
            if i in [layers // 3, 2 * layers // 3]:
                C_curr *= 2
                reduction = True
            else:
                reduction = False
            cell = Cell(
                genotype, C_prev_prev, C_prev, C_curr, reduction, reduction_prev
            )
            reduction_prev = reduction
            self.cells += [cell]
            C_prev_prev, C_prev = C_prev, cell.multiplier * C_curr
            if i == 2 * layers // 3:
                C_to_auxiliary = C_prev

        if auxiliary:
            self.auxiliary_head = AuxiliaryHeadImageNet(C_to_auxiliary, num_classes)
        self.global_pooling = nn.AvgPool2d(7)
        self.classifier = nn.Linear(C_prev, num_classes)

    def forward(self, input):
        logits_aux = None
        s0 = self.stem0(input)
        s1 = self.stem1(s0)
        for i, cell in enumerate(self.cells):
            s0, s1 = s1, cell(s0, s1, self.drop_path_prob)
            if i == 2 * self._layers // 3:
                if self._auxiliary and self.training:
                    logits_aux = self.auxiliary_head(s1)
        out = self.global_pooling(s1)
        logits = self.classifier(out.view(out.size(0), -1))
        return logits, logits_aux
