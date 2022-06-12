from torch.utils.data import DataLoader

"""
Since the torch DataLoder cannot fullfill all of our requirements, 
we desgin this DataLoader to replace it.
"""


class BaseDataLoader(DataLoader):
    def __init__(self, examples, features, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.examples = examples
        self.features = features
