class Params(object):
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

    def add(self, name: str, param):
        self.__dict__.update({name: param})

    def get(self, name: str):
        if not hasattr(self, name):
            raise ValueError(f"Attribute not found: {name}")
        return getattr(self, name)
