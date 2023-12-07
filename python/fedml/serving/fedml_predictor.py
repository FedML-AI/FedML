from abc import ABC


class FedMLPredictor(ABC):
    def __init__(self):
        if (
                type(self) is FedMLPredictor or
                (
                        type(self).predict == FedMLPredictor.predict and
                        type(self).async_predict == FedMLPredictor.async_predict
                )
        ):
            raise NotImplementedError("At least one of the predict methods must be implemented.")

    def predict(self, *args, **kwargs):
        raise NotImplementedError

    async def async_predict(self, *args, **kwargs):
        raise NotImplementedError

    def ready(self) -> bool:
        return True
