from keras.layers import Dense, Input, LSTM
from keras.models import load_model
from keras.optimizers import Adadelta
import keras.backend as K
from keras import Model
import numpy as np

from data.interface.data_manager import DataManager
from model.interface.model_manager import ModelManager


def root_mean_squared_error(y_true, y_pred):
    return K.sqrt(K.mean(K.square(y_pred - y_true), axis=-1))


class Temperature(ModelManager):
    def __init__(self, data_manager: DataManager, model_name):
        super().__init__(data_manager, model_name)

        self.x_train, self.y_train = self.data_manager.get_train_data()
        self.x_test, self.y_test = self.data_manager.get_test_data()

    def get_intermediate_output(self, layer, data):
        intermediate_layer_model = Model(inputs=self.model.input,
                                         outputs=self.model.get_layer(layer.name).output)
        return intermediate_layer_model.predict(np.expand_dims(data, axis=0))

    def load_model(self):
        self.model = load_model('models/' + self.model_name + '.h5',
                                custom_objects={'root_mean_squared_error': root_mean_squared_error})
        opt = Adadelta(lr=0.001)
        self.model.compile(optimizer=opt, loss='mean_squared_error', metrics=[root_mean_squared_error])
        self.model.summary()
        score = self.model.evaluate(self.x_test, self.y_test, verbose=0)
        print(self.model_name + "'s score : %.8f" % score[1])

    def train_model(self):
        n_hidden = 64
        n_seq = 12
        n_input = 12
        n_output = 1
        iterations = 200
        batch_size = 32

        input_layer = Input(shape=(n_seq, n_input))

        lstm1 = LSTM(n_hidden, return_sequences=True)(input_layer)
        rnn_outputs = LSTM(n_hidden, activation='tanh')(lstm1)

        rnn_outputs = Dense(n_output * 4)(rnn_outputs)
        outputs = Dense(n_output, activation='linear')(rnn_outputs)

        self.model = Model(inputs=input_layer, outputs=outputs)

        opt = Adadelta(lr=0.001)
        self.model.compile(optimizer=opt, loss='mean_squared_error', metrics=[root_mean_squared_error])

        self.model.fit(x=self.x_train, y=self.y_train, validation_data=(self.x_test, self.y_test),
                       batch_size=batch_size, epochs=iterations, shuffle=True)
        self.model.save('models/' + self.model_name + '.h5')

    def test_model(self):
        pass

    def get_layer_name(self, index):
        layer_names = [l.name for l in self.model.layers]
        return layer_names[index]

    def get_layer(self, index):
        return self.model.layers[index]

    def get_all_layer(self):
        return self.model.layers

    @staticmethod
    def __get_layer_type(layer_name):
        return layer_name.split('_')[0]

    def get_lstm_layer(self):
        indices = []
        layers = []
        for index, layer in enumerate(self.model.layers):
            if 'input' in layer.name or 'concatenate' in layer.name:
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
            if 'input' in layer.name or 'concatenate' in layer.name:
                continue
            layer_type = self.__get_layer_type(layer.name)
            if layer_type != "lstm":
                layers.append(layer)
                indices.append(index)
        return indices, layers

    def get_prob(self, data):
        data = data[np.newaxis, :]
        prob = np.squeeze(self.model.predict(data))
        return prob
