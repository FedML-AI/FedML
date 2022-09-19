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
