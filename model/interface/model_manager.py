import abc


class ModelManager(__metaclass__=abc.ABCMeta):
    @staticmethod
    def __scale(output, rmax=1, rmin=0):
        std = (output - output.min()) / (output.max() - output.min())
        return std * (rmax - rmin) + rmin

    @abc.abstractmethod
    def get_layer_name(self, index):
        pass

    @abc.abstractmethod
    def get_layer(self, index):
        pass

    @abc.abstractmethod
    def train_model(self):
        pass

    @abc.abstractmethod
    def test_model(self):
        pass

    @abc.abstractmethod
    def get_intermediate_output(self, layer, data):
        pass

    @abc.abstractmethod
    def load_model(self):
        pass

    @abc.abstractmethod
    def get_prob(self, data):
        pass
