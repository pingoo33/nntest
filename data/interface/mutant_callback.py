import abc


class MutantCallback(__metaclass__=abc.ABCMeta):
    @abc.abstractmethod
    def mutant_data(self, data):
        pass
