from abc import ABC, abstractmethod

from pyquinn.atoms.atom_graph import AtomGraph


class BackendModel(ABC):
    @abstractmethod
    def predict(self, graph: AtomGraph, **kwargs):
        pass

    @abstractmethod
    def load(self):
        pass

    @abstractmethod
    def save(self):
        pass


class Model(ABC):

    def __init__(self, backend_model: BackendModel):
        self.backend_model = backend_model

    

    


