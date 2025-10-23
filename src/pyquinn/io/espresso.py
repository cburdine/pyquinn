from pyquinn.io.data_source import DataSource

class QuantumEspresso(DataSource):

    @staticmethod
    def load_atoms(file: str | None):
        pass

    def __init__(self, work_dir: str | None):
        self.work_dir = work_dir
    
    