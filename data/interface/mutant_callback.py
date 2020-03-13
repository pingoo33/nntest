import abc


class MutantCallback(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def mutant_data(self, data):
        pass
