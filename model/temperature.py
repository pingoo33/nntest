from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, LSTM, BatchNormalization

from model.interface.custom_model import CustomModel


class Temperature(CustomModel):
    def __init__(self):
        super(Temperature, self).__init__()

        n_hidden = 64
        n_output = 1
        n_seq = 12
        n_input = 12

        self.lstm_1 = LSTM(n_hidden, return_sequences=True, input_shape=(n_seq, n_input,))
        self.lstm_2 = LSTM(n_hidden, activation='tanh')

        self.bn = BatchNormalization()
        self.dense_1 = Dense(256)
        self.dense_2 = Dense(128)
        self.dense_3 = Dense(n_output, activation='linear')

    def get_model(self):
        model = Sequential()
        model.add(self.lstm_1)
        model.add(self.lstm_2)

        model.add(self.bn)
        model.add(self.dense_1)
        model.add(self.dense_2)
        model.add(self.dense_3)

        return model
