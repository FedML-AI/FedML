"""For monkey-patching into meta-learning frameworks."""
import torch
import torch.nn.functional as F
from collections import OrderedDict
from functools import partial
import warnings

from ..consts import BENCHMARK
torch.backends.cudnn.benchmark = BENCHMARK

DEBUG = False  # Emit warning messages when patching. Use this to bootstrap new architectures.

class MetaMonkey(torch.nn.Module):
    """Trace a networks and then replace its module calls with functional calls.

    This allows for backpropagation w.r.t to weights for "normal" PyTorch networks.
    """

    def __init__(self, net):
        """Init with network."""
        super().__init__()
        self.net = net
        self.parameters = OrderedDict(net.named_parameters())


    def forward(self, inputs, parameters=None):
        """Live Patch ... :> ..."""
        # If no parameter dictionary is given, everything is normal
        if parameters is None:
            return self.net(inputs)

        # But if not ...
        param_gen = iter(parameters.values())
        method_pile = []
        counter = 0

        for name, module in self.net.named_modules():
            if isinstance(module, torch.nn.Conv2d):
                ext_weight = next(param_gen)
                if module.bias is not None:
                    ext_bias = next(param_gen)
                else:
                    ext_bias = None

                method_pile.append(module.forward)
                module.forward = partial(F.conv2d, weight=ext_weight, bias=ext_bias, stride=module.stride,
                                         padding=module.padding, dilation=module.dilation, groups=module.groups)
            elif isinstance(module, torch.nn.BatchNorm2d):
                if module.momentum is None:
                    exponential_average_factor = 0.0
                else:
                    exponential_average_factor = module.momentum

                if module.training and module.track_running_stats:
                    if module.num_batches_tracked is not None:
                        module.num_batches_tracked += 1
                        if module.momentum is None:  # use cumulative moving average
                            exponential_average_factor = 1.0 / float(module.num_batches_tracked)
                        else:  # use exponential moving average
                            exponential_average_factor = module.momentum

                ext_weight = next(param_gen)
                ext_bias = next(param_gen)
                method_pile.append(module.forward)
                module.forward = partial(F.batch_norm, running_mean=module.running_mean, running_var=module.running_var,
                                         weight=ext_weight, bias=ext_bias,
                                         training=module.training or not module.track_running_stats,
                                         momentum=exponential_average_factor, eps=module.eps)

            elif isinstance(module, torch.nn.Linear):
                lin_weights = next(param_gen)
                lin_bias = next(param_gen)
                method_pile.append(module.forward)
                module.forward = partial(F.linear, weight=lin_weights, bias=lin_bias)

            elif next(module.parameters(), None) is None:
                # Pass over modules that do not contain parameters
                pass
            elif isinstance(module, torch.nn.Sequential):
                # Pass containers
                pass
            else:
                # Warn for other containers
                if DEBUG:
                    warnings.warn(f'Patching for module {module.__class__} is not implemented.')

        output = self.net(inputs)

        # Undo Patch
        for name, module in self.net.named_modules():
            if isinstance(module, torch.nn.modules.conv.Conv2d):
                module.forward = method_pile.pop(0)
            elif isinstance(module, torch.nn.BatchNorm2d):
                module.forward = method_pile.pop(0)
            elif isinstance(module, torch.nn.Linear):
                module.forward = method_pile.pop(0)

        return output
