from abc import ABC

class FedMLPredictor(ABC):
    def __init__(self):
        if not (self.predict != FedMLPredictor.predict or self.async_predict != FedMLPredictor.async_predict):
            raise NotImplementedError("At least one of the predict methods must be implemented.")

    def predict(self, *args, **kwargs):
        raise NotImplementedError
    
    async def async_predict(self, *args, **kwargs):
        raise NotImplementedError