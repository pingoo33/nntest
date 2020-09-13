import numpy as np

from tensorflow.keras.layers import Dense, Conv2D, BatchNormalization, Activation
from tensorflow.keras.layers import AveragePooling2D, Input, Flatten
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import Model

from model.interface.model_manager import ModelManager


class Resnet(ModelManager):
    def __init__(self, model_name):
        super().__init__(model_name)

    def get_layer(self, index):
        return self.model.layers[index]

    @staticmethod
    def __lr_schedule(epoch):
        lr = 1e-3
        if epoch > 100:
            lr *= 0.5e-3
        elif epoch > 160:
            lr *= 1e-3
        elif epoch > 120:
            lr *= 1e-2
        elif epoch > 80:
            lr *= 1e-1
        return lr

    @staticmethod
    def __resnet_layer(inputs,
                       num_filters=16,
                       kernel_size=3,
                       strides=1,
                       activation='relu',
                       batch_normalization=True,
                       conv_first=True):
        conv = Conv2D(num_filters,
                      kernel_size=kernel_size,
                      strides=strides,
                      padding='same',
                      kernel_initializer='he_normal',
                      kernel_regularizer=l2(1e-4))

        x = inputs
        if conv_first:
            x = conv(x)
            if batch_normalization:
                x = BatchNormalization()(x)
            if activation is not None:
                x = Activation(activation)(x)
        else:
            if batch_normalization:
                x = BatchNormalization()(x)
            if activation is not None:
                x = Activation(activation)(x)
            x = conv(x)
        return x

    def train_model(self, x_train, y_train, x_test, y_test):
        batch_size = 32
        epochs = 200
        num_classes = 10
        num_filters_in = 16
        version = 2
        n = 3
        depth = n * 9 + 2
        num_res_blocks = int((depth - 2) / 9)
        model_type = 'Resnet%dv%d' % (depth, version)

        input_shape = x_train.shape[1:]
        inputs = Input(shape=input_shape)
        x = self.__resnet_layer(inputs=inputs, num_filters=num_filters_in, conv_first=True)

        for stage in range(3):
            for res_block in range(num_res_blocks):
                activation = 'relu'
                batch_normalization = True
                strides = 1
                if stage == 0:
                    num_filters_out = num_filters_in * 4
                    if res_block == 0:
                        activation = None
                        batch_normalization = False
                else:
                    num_filters_out = num_filters_in * 2
                    if res_block == 0:
                        strides = 2

                y = self.__resnet_layer(inputs=x,
                                        num_filters=num_filters_in,
                                        kernel_size=1,
                                        strides=strides,
                                        activation=activation,
                                        batch_normalization=batch_normalization,
                                        conv_first=False)
                y = self.__resnet_layer(inputs=y,
                                        num_filters=num_filters_in,
                                        conv_first=False)
                y = self.__resnet_layer(inputs=y,
                                        num_filters=num_filters_out,
                                        conv_first=False)
                if res_block == 0:
                    x = self.__resnet_layer(inputs=x,
                                            num_filters=num_filters_out,
                                            kernel_size=1,
                                            strides=strides,
                                            activation=None,
                                            batch_normalization=False)
                x = keras.layers.add([x, y])

            num_filters_in = num_filters_out

        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = AveragePooling2D(pool_size=8)(x)
        y = Flatten()(x)
        outputs = Dense(num_classes, activation='softmax', kernel_initializer='he_normal')(y)

        self.model = Model(inputs=inputs, outputs=outputs)
        opt = Adam(lr=self.__lr_schedule(0))
        self.model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
        self.model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_data=(x_test, y_test),
                       shuffle=True)
        self.model.save('models/' + model_type + "_" + self.model_name + '.h5')

    def kfold_train_model(self, fold_size, x_train, y_train, x_test, y_test):
        pass

    def test_model(self):
        pass

    def get_intermediate_output(self, layer, data):
        intermediate_layer_model = Model(inputs=self.model.input,
                                         outputs=self.model.get_layer(layer.name).output)
        return intermediate_layer_model.predict(np.expand_dims(data, axis=0))

    def load_model(self):
        pass

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
