from abc import ABC, abstractmethod, ABCMeta
from dataclasses import dataclass
from enum import Enum, auto
from typing import Optional, List


class GPUCardType(Enum):
    NVIDIA = auto()
    QUALCOMM = auto()
    UNKNOWN = auto()

    def __str__(self):
        return self.name


@dataclass
class GPUCard:
    id: int
    uuid: str
    name: str
    load: float
    memoryTotal: float
    memoryUsed: float
    memoryFree: float
    driver: str
    serial: Optional[str]
    display_mode: Optional[str]
    display_active: Optional[str]
    temperature: Optional[float]


class GPUTypeRegistry(type, ABC):
    GPU_TYPE_REGISTRY = {}

    def __new__(cls, name, bases, attrs):
        new_cls = type.__new__(cls, name, bases, attrs)
        cls.GPU_TYPE_REGISTRY[new_cls.__name__.lower()] = new_cls
        return new_cls

    @classmethod
    def get_gpu_utils(cls):
        return cls.GPU_TYPE_REGISTRY.values()


class GPUCardUtil(metaclass=GPUTypeRegistry):

    @classmethod
    @abstractmethod
    def detectGPUCardType(cls) -> Optional[GPUCardType]:
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def getAvailableGPUCardIDs() -> List[int]:
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def getGPUCards() -> List[GPUCard]:
        raise NotImplementedError
