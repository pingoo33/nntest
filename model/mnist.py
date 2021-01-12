from tensorflow.keras.layers import Dense, LSTM
from tensorflow.keras import Sequential

from model.interface.custom_model import CustomModel


class Mnist(CustomModel):
    def __init__(self):
        super(Mnist, self).__init__()

        n_hidden = 128
        n_output1 = 32
        n_output2 = 10
        row = 28
        col = 28

        self.lstm_1 = LSTM(n_hidden, return_sequences=True, input_shape=(row, col,))
        self.lstm_2 = LSTM(n_hidden, activation='tanh')

        self.dense_1 = Dense(n_output1, activation='relu')
        self.dense_2 = Dense(n_output2, activation='softmax')

    def call(self):
        model = Sequential()
        model.add(self.lstm_1)
        model.add(self.lstm_2)

        model.add(self.dense_1)
        model.add(self.dense_2)

        return model

