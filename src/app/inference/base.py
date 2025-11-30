from abc import ABC, abstractmethod


class BaseModel(ABC):
    @abstractmethod
    def load(self):
        """重みのロードなどを初期化"""
        ...

    @abstractmethod
    def infer(self, *args, **kwargs):
        """推論用関数"""
        ...
