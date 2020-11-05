import abc
import tensorflow.keras.backend as K


class ModelManager(metaclass=abc.ABCMeta):
    def __init__(self, model_name):
        self.model = None
        self.model_name = model_name

    def get_gradients_function(self, layer_name=None):
        nodes = [layer.input for layer in self.model.layers if layer.name == layer_name or layer_name is None]
        nodes_names = [n.name for n in nodes]
        if self.model.optimizer is None:
            raise Exception('Please compile the model first. The loss function is required to compute the gradients.')
        grads = self.model.optimizer.get_gradients(self.model.total_loss, nodes)
        symb_inputs = (self.model._feed_inputs + self.model._feed_targets + self.model._feed_sample_weights)
        f = K.function(symb_inputs, grads)
        return f, nodes_names

    def cal_gradient(self, f, x, y, nodes_names):
        x_, y_, sample_weight_ = self.model._standardize_user_data(x, y)
        gradients_values = f(x_ + y_ + sample_weight_)
        result = dict(zip(nodes_names, gradients_values))
        return result

    @abc.abstractmethod
    def get_layer(self, index):
        pass

    @abc.abstractmethod
    def train_model(self, x_train, y_train, x_test, y_test):
        pass

    @abc.abstractmethod
    def kfold_train_model(self, fold_size, x_train, y_train, x_test, y_test):
        pass

    @abc.abstractmethod
    def test_model(self, test_x, test_y):
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

    @abc.abstractmethod
    def get_lstm_layer(self):
        pass

    @abc.abstractmethod
    def get_fc_layer(self):
        pass
