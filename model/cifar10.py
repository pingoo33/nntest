import numpy as np
from sklearn.model_selection import KFold

from tensorflow.keras.layers import Dense, Input, Activation, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras import Model

from model.interface.model_manager import ModelManager


class Cifar10(ModelManager):
    def __init__(self, model_name):
        super().__init__(model_name)

    def get_layer(self, index):
        return self.model.layers[index]

    def train_model(self, x_train, y_train, x_test, y_test):
        n_output = 512
        num_classes = 10
        epochs = 100
        batch_size = 32

        input_layer = Input(shape=x_train.shape[1:])
        conv1 = Conv2D(32, (3, 3), padding='same')(input_layer)
        acti1 = Activation('relu')(conv1)
        max1 = MaxPooling2D(pool_size=(2, 2))(acti1)

        conv2 = Conv2D(64, (3, 3), padding='same')(max1)
        acti2 = Activation('relu')(conv2)
        conv3 = Conv2D(64, (3, 3))(acti2)
        acti3 = Activation('relu')(conv3)
        max2 = MaxPooling2D(pool_size=(2, 2))(acti3)

        flatten = Flatten()(max2)
        outputs = Dense(n_output, activation='relu')(flatten)
        outputs = Dense(num_classes, activation='softmax')(outputs)

        self.model = Model(inputs=input_layer, outputs=outputs)

        opt = RMSprop(lr=0.0001, decay=1e-6)
        self.model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
        self.model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_data=(x_test, y_test),
                       shuffle=True)

        self.model.save('models/%s.h5' % self.model_name)

    def kfold_train_model(self, fold_size, x_train, y_train, x_test, y_test):
        n_output = 512
        num_classes = 10
        epochs = 100
        batch_size = 32

        model_list = []
        accuracy_list = []

        kfold = KFold(n_splits=fold_size, shuffle=True)
        for train_index, test_index in kfold.split(x_train, y_train):
            input_layer = Input(shape=x_train.shape[1:])
            conv1 = Conv2D(32, (3, 3), padding='same')(input_layer)
            acti1 = Activation('relu')(conv1)
            max1 = MaxPooling2D(pool_size=(2, 2))(acti1)

            conv2 = Conv2D(64, (3, 3), padding='same')(max1)
            acti2 = Activation('relu')(conv2)
            conv3 = Conv2D(64, (3, 3))(acti2)
            acti3 = Activation('relu')(conv3)
            max2 = MaxPooling2D(pool_size=(2, 2))(acti3)

            flatten = Flatten()(max2)
            outputs = Dense(n_output, activation='relu')(flatten)
            outputs = Dense(num_classes, activation='softmax')(outputs)

            model = Model(inputs=input_layer, outputs=outputs)

            opt = RMSprop(lr=0.0001, decay=1e-6)
            model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
            model.fit(x_train[train_index], y_train[train_index], batch_size=batch_size, epochs=epochs,
                      validation_data=(x_train[test_index], y_train[test_index]), shuffle=True)
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
        opt = RMSprop(lr=0.0001, decay=1e-6)
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
