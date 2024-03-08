class Singleton(type):

    """
    We are not initializing the singleton objects using the `fedml.core.common.singleton`,
    because that Singleton approach does not allow to pass arguments during initialization.
    In particular, the error that is raised with the previous approach is:
    `TypeError: object.__new__() takes exactly one argument (the type to instantiate)`
    """

    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]
