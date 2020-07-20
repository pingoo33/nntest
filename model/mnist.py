from keras.layers import Dense, Input, LSTM
from keras.models import load_model, Sequential
from keras.optimizers import Adam
from keras import Model
import numpy as np
from sklearn.model_selection import KFold

from model.interface.model_manager import ModelManager


class Mnist(ModelManager):
    def __init__(self, model_name):
        super().__init__(model_name)

    def get_layer(self, index):
        return self.model.layers[index]

    def train_model(self, x_train, y_train, x_test, y_test):
        n_hidden = 128
        n_output1 = 32
        n_output2 = 10
        epochs = 3
        row, col = x_train.shape[1:]

        input_layer = Input(shape=(row, col))

        lstm1 = LSTM(n_hidden, return_sequences=True)(input_layer)
        rnn_outputs = LSTM(n_hidden, activation='tanh')(lstm1)

        outputs = Dense(n_output1, activation='relu')(rnn_outputs)
        outputs = Dense(n_output2, activation='softmax')(outputs)

        model = Model(inputs=input_layer, outputs=outputs)

        opt = Adam(lr=1e-3, decay=1e-5)
        model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
        model.fit(x_train, y_train, epochs=epochs, validation_data=(x_test, y_test))

        model.save('models/%s.h5' % self.model_name)

    def kfold_train_model(self, fold_size, x_train, y_train, x_test, y_test):
        n_hidden = 128
        n_output1 = 32
        n_output2 = 10
        epochs = 3
        row, col = x_train.shape[1:]

        model_list = []
        accuracy_list = []

        kfold = KFold(n_splits=fold_size, shuffle=True)
        for train_index, test_index in kfold.split(x_train, y_train):
            input_layer = Input(shape=(row, col))

            lstm1 = LSTM(n_hidden, return_sequences=True)(input_layer)
            rnn_outputs = LSTM(n_hidden, activation='tanh')(lstm1)

            outputs = Dense(n_output1, activation='relu')(rnn_outputs)
            outputs = Dense(n_output2, activation='softmax')(outputs)

            model = Model(inputs=input_layer, outputs=outputs)
            opt = Adam(lr=1e-3, decay=1e-5)
            model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
            model.fit(x_train[train_index], y_train[train_index], epochs=epochs,
                      validation_data=(x_train[test_index], y_train[test_index]))
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

    def get_intermediate_output(self, layer, data):
        intermediate_layer_model = Model(inputs=self.model.input,
                                         outputs=self.model.get_layer(layer.name).output)
        return intermediate_layer_model.predict(np.expand_dims(data, axis=0))

    def load_model(self):
        self.model = load_model('models/%s.h5' % self.model_name)
        opt = Adam(lr=1e-3, decay=1e-5)
        self.model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
        self.model.summary()

    def get_prob(self, data):
        data = data[np.newaxis, :]
        prob = np.squeeze(self.model.predict(data))
        return prob

    @staticmethod
    def __get_layer_type(layer_name):
        return layer_name.split('_')[0]

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
