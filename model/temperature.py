from keras.layers import Dense, Input, LSTM
from keras.models import load_model
from keras.optimizers import Adadelta
import keras.backend as K
from keras import Model
import numpy as np
from sklearn.model_selection import KFold

from model.interface.model_manager import ModelManager


def root_mean_squared_error(y_true, y_pred):
    return K.sqrt(K.mean(K.square(y_pred - y_true), axis=-1))


class Temperature(ModelManager):
    def __init__(self, model_name):
        super().__init__(model_name)

    def get_intermediate_output(self, layer, data):
        intermediate_layer_model = Model(inputs=self.model.input,
                                         outputs=self.model.get_layer(layer.name).output)
        return intermediate_layer_model.predict(np.expand_dims(data, axis=0))

    def load_model(self):
        self.model = load_model('models/' + self.model_name + '.h5',
                                custom_objects={'root_mean_squared_error': root_mean_squared_error}, compile=False)
        opt = Adadelta(lr=0.001)
        self.model.compile(optimizer=opt, loss='mean_squared_error', metrics=[root_mean_squared_error])
        self.model.summary()

    def train_model(self, x_train, y_train, x_test, y_test):
        n_hidden = 64
        n_seq = 12
        n_input = 12
        n_output = 1
        epochs = 200
        batch_size = 32

        input_layer = Input(shape=(n_seq, n_input))

        lstm1 = LSTM(n_hidden, return_sequences=True)(input_layer)
        rnn_outputs = LSTM(n_hidden, activation='tanh')(lstm1)

        outputs = Dense(n_output * 4)(rnn_outputs)
        outputs = Dense(n_output, activation='linear')(outputs)

        model = Model(inputs=input_layer, outputs=outputs)

        opt = Adadelta(lr=0.001)
        model.compile(optimizer=opt, loss='mean_squared_error', metrics=[root_mean_squared_error])

        model.fit(x=x_train, y=y_train,
                  validation_data=(x_test, y_test),
                  batch_size=batch_size, epochs=epochs, shuffle=True)

        model.save('models/%s.h5' % self.model_name)

    def kfold_train_model(self, fold_size, x_train, y_train, x_test, y_test):
        n_hidden = 64
        n_seq = 12
        n_input = 12
        n_output = 1
        epochs = 200
        batch_size = 32

        model_list = []
        accuracy_list = []

        kfold = KFold(n_splits=fold_size, shuffle=True)
        for train_index, test_index in kfold.split(x_train, y_train):
            input_layer = Input(shape=(n_seq, n_input))

            lstm1 = LSTM(n_hidden, return_sequences=True)(input_layer)
            rnn_outputs = LSTM(n_hidden, activation='tanh')(lstm1)

            rnn_outputs = Dense(n_output * 4)(rnn_outputs)
            outputs = Dense(n_output, activation='linear')(rnn_outputs)

            model = Model(inputs=input_layer, outputs=outputs)

            opt = Adadelta(lr=0.001)
            model.compile(optimizer=opt, loss='mean_squared_error', metrics=[root_mean_squared_error])

            model.fit(x=x_train[train_index], y=y_train[train_index],
                      validation_data=(x_train[test_index], y_train[test_index]),
                      batch_size=batch_size, epochs=epochs, shuffle=True)
            model_list.append(model)
            accuracy_list.append(model.evaluate(x_test, y_test)[1])

        acc = 0.0
        for i in range(len(model_list)):
            if accuracy_list[i] > acc:
                self.model = model_list[i]
                acc = accuracy_list[i]

        print("accuracy : %s" % str(acc))
        self.model.save('models/%s.h5' % self.model_name)

    def test_model(self):
        pass

    def get_layer(self, index):
        return self.model.layers[index]

    @staticmethod
    def __get_layer_type(layer_name):
        return layer_name.split('_')[0]

    def get_lstm_layer(self):
        indices = []
        layers = []
        for index, layer in enumerate(self.model.layers):
            if 'input' in layer.name or 'concatenate' in layer.name or index == len(self.model.layers) - 1:
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
            if 'input' in layer.name or 'concatenate' in layer.name or index == len(self.model.layers) - 1:
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
