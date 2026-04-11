from .resnet18 import ResNet18
from .resnet20 import ResNet20
from .simple_cnn import SimpleCNN
from .lenet5 import LeNet5


MODEL_REGISTRY = {
    "SimpleCNN": SimpleCNN,
    "ResNet20": ResNet20,
    "ResNet18": ResNet18,
    "LeNet5": LeNet5,
}


def get_model_class(model_name: str):
    if model_name not in MODEL_REGISTRY:
        raise ValueError(
            f"Unknown model '{model_name}'. Available: {list(MODEL_REGISTRY.keys())}"
        )
    return MODEL_REGISTRY[model_name]


def create_model(args):
    model_name = getattr(args, "model", "SimpleCNN")
    model_cls = get_model_class(model_name)
    return model_cls()
