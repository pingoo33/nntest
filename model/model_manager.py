from tensorflow.keras.models import load_model
from tensorflow.keras import Model
import keras.backend as K
import numpy as np
from sklearn.model_selection import KFold
import keract

from model.interface.model_manager import ModelManager
from model.interface.train_option import TrainOption


class ModelManagerImpl(ModelManager):
    def __init__(self, model: Model, model_name, train_opt: TrainOption):
        super().__init__(model, model_name, train_opt)

    def get_gradients(self, x, y=None, layer_name=None):
        gradients = keract.get_gradients_of_activations(self.model, x, y,
                                                        layer_names=layer_name)
        return np.squeeze(list(gradients.values())[0])

    def get_activations(self, x, layer_name=None):
        nodes = [layer.output for layer in self.model.layers
                 if layer.name == layer_name or layer_name is None]
        input_layer_outputs, layer_outputs = [], []
        [input_layer_outputs.append(node) if 'input_' in node.name else layer_outputs.append(node) for node in nodes]
        activations = keract.get_activations(self.model, x,
                                             nodes_to_evaluate=layer_outputs)
        return np.squeeze(list(activations.values()))

    def get_layer(self, index):
        return self.model.layers[index]

    def train_model(self, x_train, y_train, x_test, y_test):
        self.model.compile(optimizer=self.train_opt.opt, loss=self.train_opt.loss, metrics=self.train_opt.metrics)
        self.model.fit(x_train, y_train, batch_size=self.train_opt.batch_size, epochs=self.train_opt.epochs,
                       validation_data=(x_test, y_test),
                       shuffle=True)

        self.model.save('models/%s.h5' % self.model_name, save_format='tf')

    def kfold_train_model(self, fold_size, x_train, y_train, x_test, y_test):
        models = []
        results = []

        kfold = KFold(n_splits=fold_size, shuffle=True)
        for train_i, test_i in kfold.split(x_train, y_train):
            self.model.compile(optimizer=self.train_opt.opt, loss=self.train_opt.loss, metrics=self.train_opt.metrics)
            self.model.fit(x_train[train_i], y_train[train_i],
                           batch_size=self.train_opt.batch_size, epochs=self.train_opt.epochs,
                           validation_data=(x_test[test_i], y_test[test_i]),
                           shuffle=True)
            models.append(self.model)
            results.append(self.model.evaluate(x_test, y_test)[1])

        acc = 0.0
        for i in range(len(models)):
            if results[i] > acc:
                self.model = models[i]
                acc = results[i]

        self.model.save('models/%s.h5' % self.model_name, save_format='tf')

    def test_model(self, test_x, test_y):
        pass

    def get_intermediate_output(self, layer, data):
        intermediate_layer_model = Model(inputs=self.model.input,
                                         outputs=self.model.get_layer(layer.name).output)
        return intermediate_layer_model.predict(np.expand_dims(data, axis=0))

    def load_model(self):
        self.model = load_model('models/%s.h5' % self.model_name)
        self.model.compile(optimizer=self.train_opt.opt, loss=self.train_opt.loss, metrics=self.train_opt.metrics)
        self.model.summary()

    def get_prob(self, data):
        data = data[np.newaxis, :]
        prob = np.squeeze(self.model.predict(data))
        return prob

    def get_lstm_layer(self):
        indices = []
        layers = []
        for index, layer in enumerate(self.model.layers):
            if 'input' in layer.name \
                    or 'concatenate' in layer.name \
                    or index == len(self.model.layers) - 1 \
                    or 'flatten' in layer.name:
                continue
            layer_type = self.__get_layer_type(layer.name)
            if layer_type == "lstm":
                layers.append(layer)
                indices.append(index)
        return indices, layers

    def get_fc_layer(self):
        indices = []
        layers = []
        for index, layer in enumerate(self.model.layers):
            if 'input' in layer.name \
                    or 'concatenate' in layer.name \
                    or index == len(self.model.layers) - 1 \
                    or 'flatten' in layer.name:
                continue
            layer_type = self.__get_layer_type(layer.name)
            if layer_type != "lstm":
                layers.append(layer)
                indices.append(index)
        return indices, layers

    @staticmethod
    def __get_layer_type(layer_name):
        return layer_name.split('_')[0]
