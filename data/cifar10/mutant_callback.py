from data.interface.mutant_callback import MutantCallback
from model.interface.model_manager import ModelManager


class Cifar10MutantCallback(MutantCallback):
    def __init__(self, model_manager: ModelManager):
        self.model_manager = model_manager

    def mutant_data(self, data):
        pass
