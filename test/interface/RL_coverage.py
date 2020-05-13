import abc
from .coverage import Coverage


class RLCoverage(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def get_activation(self):
        pass

    @abc.abstractmethod
    def calculate_coverage(self):
        pass

    @abc.abstractmethod
    def update_features(self, data):
        pass

    @abc.abstractmethod
    def update_graph(self, num_samples):
        pass

    @staticmethod
    @abc.abstractmethod
    def calculate_variation(data):
        pass

    @abc.abstractmethod
    def update_frequency_graph(self):
        pass

    @abc.abstractmethod
    def display_graph(self):
        pass

    @abc.abstractmethod
    def display_frequency_graph(self):
        pass

    @abc.abstractmethod
    def display_stat(self):
        pass
