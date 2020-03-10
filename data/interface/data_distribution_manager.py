import abc


class DataDistribution(__metaclass__=abc.ABCMeta):
    @abc.abstractmethod
    def load_distribution(self):
        pass

    @abc.abstractmethod
    def get_distribution(self, data):
        pass
