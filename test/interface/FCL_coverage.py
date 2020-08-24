import abc
from keras import backend as K
from .coverage import Coverage


class FCLCoverage(metaclass=abc.ABCMeta):
    @staticmethod
    def __normalize(x):
        return x / (K.sqrt(K.mean(K.square(x))) + 1e-5)

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

    @abc.abstractmethod
    def get_name(self):
        pass
