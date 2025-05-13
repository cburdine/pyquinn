from abc import ABC, abstractmethod


class ModelBase:
    """ Base class for Model instances"""


class Model(ABC, ModelBase):
    """ Abstract class for the basic PyQuINN Model API"""

    @abstractmethod
    def predict(self, **kwargs):
        pass

    @abstractmethod
    def load(self, **kwargs):
        pass

    @abstractmethod
    def save(self, **kwargs):
        pass
