"""Optimization setups."""

from dataclasses import dataclass


def training_strategy(strategy, lr=None, epochs=None, dryrun=False):
    """Parse training strategy."""
    if strategy == 'conservative':
        defs = ConservativeStrategy(lr, epochs, dryrun)
    elif strategy == 'adam':
        defs = AdamStrategy(lr, epochs, dryrun)
    else:
        raise ValueError('Unknown training strategy.')
    return defs


@dataclass
class Strategy:
    """Default usual parameters, not intended for parsing."""

    epochs : int
    batch_size : int
    optimizer : str
    lr : float
    scheduler : str
    weight_decay : float
    validate : int
    warmup: bool
    dryrun : bool
    dropout : float
    augmentations : bool

    def __init__(self, lr=None, epochs=None, dryrun=False):
        """Defaulted parameters. Apply overwrites from args."""
        if epochs is not None:
            self.epochs = epochs
        if lr is not None:
            self.lr = lr
        if dryrun:
            self.dryrun = dryrun
        self.validate = 10

@dataclass
class ConservativeStrategy(Strategy):
    """Default usual parameters, defines a config object."""

    def __init__(self, lr=None, epochs=None, dryrun=False):
        """Initialize training hyperparameters."""
        self.lr = 0.1
        self.epochs = 120
        self.batch_size = 128
        self.optimizer = 'SGD'
        self.scheduler = 'linear'
        self.warmup = False
        self.weight_decay : float = 5e-4
        self.dropout = 0.0
        self.augmentations = True
        self.dryrun = False
        super().__init__(lr=None, epochs=None, dryrun=False)


@dataclass
class AdamStrategy(Strategy):
    """Start slowly. Use a tame Adam."""

    def __init__(self, lr=None, epochs=None, dryrun=False):
        """Initialize training hyperparameters."""
        self.lr = 1e-3 / 10
        self.epochs = 120
        self.batch_size = 32
        self.optimizer = 'AdamW'
        self.scheduler = 'linear'
        self.warmup = True
        self.weight_decay : float = 5e-4
        self.dropout = 0.0
        self.augmentations = True
        self.dryrun = False
        super().__init__(lr=None, epochs=None, dryrun=False)
