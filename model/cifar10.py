from tensorflow.keras.layers import Dense, Activation, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras import Sequential

from model.interface.custom_model import CustomModel


class Cifar10(CustomModel):
    def __init__(self):
        super(Cifar10, self).__init__()

        n_output = 512
        num_classes = 10
        row = 32
        col = 32

        self.conv_1 = Conv2D(32, (3, 3), padding='same', input_shape=(row, col,))
        self.act_1 = Activation('relu')
        self.max_1 = MaxPooling2D(pool_size=(2, 2))

        self.conv_2 = Conv2D(64, (3, 3), padding='same')
        self.conv_3 = Conv2D(64, (3, 3))

        self.flatten = Flatten()
        self.dense_1 = Dense(n_output, activation='relu')
        self.dense_2 = Dense(num_classes, activation='softmax')

    def get_model(self):
        model = Sequential()
        model.add(self.conv_1)
        model.add(self.act_1)
        model.add(self.max_1)

        model.add(self.conv_2)
        model.add(self.act_1)
        model.add(self.conv_3)
        model.add(self.act_1)
        model.add(self.max_1)

        model.add(self.flatten)
        model.add(self.dense_1)
        model.add(self.dense_2)

        return model


