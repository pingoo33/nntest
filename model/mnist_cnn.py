from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten
from tensorflow.keras import Sequential

from model.interface.custom_model import CustomModel


class MnistCNN(CustomModel):
    def __init__(self):
        super(MnistCNN, self).__init__()

        n_output1 = 128
        n_output2 = 10
        row = 28
        col = 28

        self.conv_1 = Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(row, col, 1,))
        self.conv_2 = Conv2D(64, kernel_size=(3, 3), activation='relu')
        self.max_1 = MaxPooling2D(pool_size=(2, 2))

        self.flatten = Flatten()
        self.dense_1 = Dense(n_output1, activation='relu')
        self.dense_2 = Dense(n_output2, activation='softmax')

    def get_model(self):
        model = Sequential()
        model.add(self.conv_1)
        model.add(self.conv_2)
        model.add(self.max_1)

        model.add(self.flatten)
        model.add(self.dense_1)
        model.add(self.dense_2)

        return model


