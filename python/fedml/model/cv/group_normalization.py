import torch.nn.functional as F
from torch.nn.modules.batchnorm import _BatchNorm

"""Pytorch implementation of group normalization in https://arxiv.org/abs/1803.08494 (Following the PyTorch Style)"""


def group_norm(
    input,
    group,
    running_mean,
    running_var,
    weight=None,
    bias=None,
    use_input_stats=True,
    momentum=0.1,
    eps=1e-5,
):
    """
    Applies Group Normalization for channels in the same group in each data sample in a batch.

    Args:
        input (Tensor): The input tensor of shape (N, C, *), where N is the batch size,
            C is the number of channels, and * represents any number of additional dimensions.
        group (int): The number of groups to divide the channels into.
        running_mean (Tensor or None): A tensor of running means for each group, typically
            from previous batches. Set to None if `use_input_stats` is True.
        running_var (Tensor or None): A tensor of running variances for each group, typically
            from previous batches. Set to None if `use_input_stats` is True.
        weight (Tensor or None): A tensor to scale the normalized values for each channel.
        bias (Tensor or None): A tensor to add an offset to the normalized values for each channel.
        use_input_stats (bool): If True, batch statistics (mean and variance) are computed
            from the input tensor for normalization. If False, `running_mean` and `running_var`
            are used for normalization.
        momentum (float): The momentum factor for updating running statistics.
        eps (float): A small value added to the denominator for numerical stability.

    Returns:
        Tensor: The normalized output tensor with the same shape as the input.

    Note:
        Group Normalization is applied to the channels of the input tensor separately within each group.
        If `use_input_stats` is True, running statistics (mean and variance) will not be used for
        normalization, and batch statistics will be computed from the input tensor.

    See Also:
        - :class:`~torch.nn.GroupNorm1d` for 1D input (sequence data).
        - :class:`~torch.nn.GroupNorm2d` for 2D input (image data).
        - :class:`~torch.nn.GroupNorm3d` for 3D input (volumetric data).
    """

    if not use_input_stats and (running_mean is None or running_var is None):
        raise ValueError(
            "Expected running_mean and running_var to be not None when use_input_stats=False"
        )

    b, c = input.size(0), input.size(1)
    if weight is not None:
        weight = weight.repeat(b)
    if bias is not None:
        bias = bias.repeat(b)

    def _instance_norm(
        input,
        group,
        running_mean=None,
        running_var=None,
        weight=None,
        bias=None,
        use_input_stats=None,
        momentum=None,
        eps=None,
    ):
        """
        Applies Instance Normalization for channels within each group in the input tensor.

        Args:
            input (Tensor): The input tensor of shape (N, C, *), where N is the batch size,
                C is the number of channels, and * represents any number of additional dimensions.
            group (int): The number of groups to divide the channels into.
            running_mean (Tensor or None): A tensor of running means for each group, typically
                from previous batches. Set to None if `use_input_stats` is True.
            running_var (Tensor or None): A tensor of running variances for each group, typically
                from previous batches. Set to None if `use_input_stats` is True.
            weight (Tensor or None): A tensor to scale the normalized values for each channel.
            bias (Tensor or None): A tensor to add an offset to the normalized values for each channel.
            use_input_stats (bool or None): If True, batch statistics (mean and variance) are computed
                from the input tensor for normalization. If False, `running_mean` and `running_var`
                are used for normalization. If None, it defaults to True during training and False during inference.
            momentum (float): The momentum factor for updating running statistics.
            eps (float): A small value added to the denominator for numerical stability.

        Returns:
            Tensor: The normalized output tensor with the same shape as the input.

        Note:
            Instance Normalization is applied to the channels of the input tensor separately within each group.
            If `use_input_stats` is True, running statistics (mean and variance) will not be used for
            normalization, and batch statistics will be computed from the input tensor.

        See Also:
            - :class:`~torch.nn.InstanceNorm1d` for 1D input (sequence data).
            - :class:`~torch.nn.InstanceNorm2d` for 2D input (image data).
            - :class:`~torch.nn.InstanceNorm3d` for 3D input (volumetric data).
        """
        # Repeat stored stats and affine transform params if necessary
        if running_mean is not None:
            running_mean_orig = running_mean
            running_mean = running_mean_orig.repeat(b)
        if running_var is not None:
            running_var_orig = running_var
            running_var = running_var_orig.repeat(b)

        # norm_shape = [1, b * c / group, group]
        # print(norm_shape)
        # Apply instance norm
        input_reshaped = input.contiguous().view(
            1, int(b * c / group), group, *input.size()[2:]
        )

        out = F.batch_norm(
            input_reshaped,
            running_mean,
            running_var,
            weight=weight,
            bias=bias,
            training=use_input_stats,
            momentum=momentum,
            eps=eps,
        )

        # Reshape back
        if running_mean is not None:
            running_mean_orig.copy_(
                running_mean.view(b, int(c / group)).mean(0, keepdim=False)
            )
        if running_var is not None:
            running_var_orig.copy_(
                running_var.view(b, int(c / group)).mean(0, keepdim=False)
            )

        return out.view(b, c, *input.size()[2:])

    return _instance_norm(
        input,
        group,
        running_mean=running_mean,
        running_var=running_var,
        weight=weight,
        bias=bias,
        use_input_stats=use_input_stats,
        momentum=momentum,
        eps=eps,
    )


class _GroupNorm(_BatchNorm):
    """
    Applies Group Normalization over a mini-batch of inputs.

    Group Normalization divides the channels into groups and computes statistics
    (mean and variance) separately for each group, normalizing each group independently.
    It can be used as a normalization layer in various neural network architectures.

    Args:
        num_features (int): Number of channels in the input tensor.
        num_groups (int): Number of groups to divide the channels into.
        eps (float): A small value added to the denominator for numerical stability.
        momentum (float): The momentum factor for updating running statistics.
        affine (bool): If True, learnable affine parameters (weight and bias) are applied to
            the normalized output. Default is False.
        track_running_stats (bool): If True, running statistics (mean and variance) are tracked
            during training. Default is False.

    Attributes:
        num_groups (int): Number of groups the channels are divided into.
        track_running_stats (bool): If True, running statistics (mean and variance) are tracked
            during training.

    Note:
        The input tensor should have shape (N, C, *), where N is the batch size, C is the
        number of channels, and * represents any number of additional dimensions.

    See Also:
        - :class:`~torch.nn.GroupNorm` for a user-friendly interface.
        - :class:`~torch.nn.BatchNorm2d` for standard Batch Normalization.
    """
    def __init__(
        self,
        num_features,
        num_groups=1,
        eps=1e-5,
        momentum=0.1,
        affine=False,
        track_running_stats=False,
    ):
        self.num_groups = num_groups
        self.track_running_stats = track_running_stats
        super(_GroupNorm, self).__init__(
            int(num_features / num_groups), eps, momentum, affine, track_running_stats
        )

    def _check_input_dim(self, input):
        return NotImplemented

    def forward(self, input):
        self._check_input_dim(input)

        return group_norm(
            input,
            self.num_groups,
            self.running_mean,
            self.running_var,
            self.weight,
            self.bias,
            self.training or not self.track_running_stats,
            self.momentum,
            self.eps,
        )


class GroupNorm2d(_GroupNorm):
    """Applies Group Normalization over a 4D input (a mini-batch of 2D inputs
    with an additional channel dimension) as described in the paper
    "Group Normalization" (https://arxiv.org/pdf/1803.08494.pdf).
    
    Args:
        num_features (int): Number of channels in the input tensor.
        num_groups (int): Number of groups to divide the channels into.
        eps (float): A small value added to the denominator for numerical stability. 
            Default: 1e-5.
        momentum (float): The value used for computing running statistics (mean and variance).
            Default: 0.1.
        affine (bool): If True, learnable affine parameters (weight and bias) are applied to
            the normalized output. Default: True.
        track_running_stats (bool): If True, this module tracks running statistics
            (mean and variance) during training. If False, it uses batch statistics in both
            training and evaluation modes. Default: False.

    Shape:
        - Input: (N, C, H, W)
        - Output: (N, C, H, W) (same shape as input)

    Examples:
        >>> # Without Learnable Parameters
        >>> m = GroupNorm2d(100, 4)
        >>> # With Learnable Parameters
        >>> m = GroupNorm2d(100, 4, affine=True)
        >>> input = torch.randn(20, 100, 35, 45)
        >>> output = m(input)

    Note:
        The input tensor should have shape (N, C, H, W), where N is the batch size,
        C is the number of channels, H is the height, and W is the width.

    See Also:
        - :class:`~torch.nn.GroupNorm` for a user-friendly interface.
        - :class:`~torch.nn.BatchNorm2d` for standard Batch Normalization for 2D data.
    """


    def _check_input_dim(self, input):
        if input.dim() != 4:
            raise ValueError("expected 4D input (got {}D input)".format(input.dim()))


class GroupNorm3d(_GroupNorm):
    """
    Applies 3D Group Normalization over a mini-batch of 3D inputs.

    Group Normalization divides the channels into groups and computes statistics
    (mean and variance) separately for each group, normalizing each group independently.
    It is designed for 3D data with the format (B, C, D, H, W), where B is the batch size,
    C is the number of channels, D is the depth, H is the height, and W is the width.

    Args:
        num_features (int): Number of channels in the input tensor.
        num_groups (int): Number of groups to divide the channels into.
        eps (float): A small value added to the denominator for numerical stability.
        momentum (float): The momentum factor for updating running statistics.
        affine (bool): If True, learnable affine parameters (weight and bias) are applied to
            the normalized output. Default is False.
        track_running_stats (bool): If True, running statistics (mean and variance) are tracked
            during training. Default is False.

    Attributes:
        num_groups (int): Number of groups the channels are divided into.
        track_running_stats (bool): If True, running statistics (mean and variance) are tracked
            during training.

    Note:
        The input tensor should have shape (N, C, D, H, W), where N is the batch size, C is the
        number of channels, D is the depth, H is the height, and W is the width.

    See Also:
        - :class:`~torch.nn.GroupNorm` for a user-friendly interface.
        - :class:`~torch.nn.BatchNorm3d` for standard Batch Normalization for 3D data.
    """

    def _check_input_dim(self, input):
        if input.dim() != 5:
            raise ValueError("expected 5D input (got {}D input)".format(input.dim()))
