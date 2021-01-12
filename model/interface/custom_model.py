import abc


class CustomModel(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def get_model(self):
        pass