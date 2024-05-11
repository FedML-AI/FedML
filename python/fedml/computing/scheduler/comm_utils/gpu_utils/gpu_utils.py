from abc import ABC, abstractmethod
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


class GPUCardUtil(ABC):

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
