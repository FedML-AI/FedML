import torch

class EnsembleModel(torch.nn.Module):
    def __init__(self, model, ema):
        super(EnsembleModel, self).__init__()
        self.model = model
        self.ema = ema
