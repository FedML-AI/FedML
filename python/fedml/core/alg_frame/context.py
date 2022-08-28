from .params import Params
from ..common.singleton import Singleton


class Context(Params, Singleton):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)