import abc


class Oracle(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def pass_oracle(self, src, dest):
        pass

    @abc.abstractmethod
    def measure(self, src, dest):
        pass