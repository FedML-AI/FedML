class Params(object):
    KEY_MODEL_PARAMS = "model_params"
    """
    Unified Parameter Object for passing arguments among APIs
            from the algorithm frame (e.g., client_trainer.py and server aggregator.py).

    Usage::
        >> my_params = Params()
        >> # add parameter
        >> my_params.add(name="w", param=model_weights)
        >> # get parameter
        >> my_params.w
    """

    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

    def add(self, name: str, value):
        self.__dict__[name] = value

    def get(self, name: str):
        if not hasattr(self, name):
            return None
        return getattr(self, name)

    def keys(self):
        return self.__dict__.keys()

    def values(self):
        return self.__dict__.values()

    def has(self, name: str):
        return hasattr(self, name)

    def __contains__(self, name: str):
        return name in self.__dict__

    def add_dict(self, new_values):
        for key, value in new_values.items():
            self.add(key, value)

    def __getitem__(self, name: str):
        return getattr(self, name)

    def pop(self, name):
        self.__dict__.pop(name)

