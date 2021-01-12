import numpy as np
from tensorflow.keras.models import Model
import keras.backend as K
from tensorflow.keras.layers import Input, LSTM

from model.interface.model_manager import ModelManager


class StateManager:
    def __init__(self, model_manager: ModelManager, layer_index):
        self.model_manager = model_manager
        self.layer_index = layer_index

    def aggregate_inf(self, data_set, seq):
        h_train = self.get_hidden_state(data_set)
        t1 = int(seq[0])
        t2 = int(seq[1])
        indices = slice(t1, t2 + 1)

        alpha1 = np.sum(np.where(h_train > 0, h_train, 0), axis=2)
        alpha2 = np.sum(np.where(h_train < 0, h_train, 0), axis=2)

        alpha11 = np.insert(np.delete(alpha1, -1, axis=1), 0, 0, axis=1)
        alpha22 = np.insert(np.delete(alpha2, -1, axis=1), 0, 0, axis=1)

        alpha_TC = np.abs(alpha1 + alpha2)
        alpha_SC = np.abs(alpha1 - alpha11 + alpha2 - alpha22)

        mean_TC = np.mean(alpha_TC[:, indices])
        std_TC = np.std(alpha_TC[:, indices])

        max_BC = np.max(alpha_TC)
        min_BC = np.min(alpha_TC)

        max_SC = np.max(alpha_SC)
        min_SC = np.min(alpha_SC)

        return mean_TC, std_TC, max_SC, min_SC, max_BC, min_BC

    @staticmethod
    def __hard_sigmoid(x):
        return np.maximum(0, np.minimum(1, 0.2 * x + 0.5))

    def cal_hidden_keras(self, test, layer_num):
        if layer_num == 0:
            acx = test
        else:
            acx = self.model_manager.get_activations(np.array(test),
                                                     layer_name=self.model_manager.model.layers[layer_num - 1].name)

        units = int(int(self.model_manager.model.layers[layer_num].trainable_weights[0].shape[1]) / 4)

        if len(np.shape(acx)) < len(np.shape(test)):
            acx = np.array([acx])
        elif len(np.shape(acx)) > len(np.shape(test)):
            acx = acx[0]

        inp = Input(batch_shape=(None, acx.shape[-2], acx.shape[-1]), name="input")
        rnn, s, c = LSTM(units,
                         return_sequences=True,
                         stateful=False,
                         return_state=True,
                         name="RNN")(inp)
        states = Model(inputs=[inp], outputs=[s, c, rnn])

        for layer in states.layers:
            if layer.name == "RNN":
                layer.set_weights(self.model_manager.model.layers[layer_num].get_weights())

        h_t, c_t, rnn = states.predict(acx)

        return rnn

    def get_hidden_state(self, data):
        return self.cal_hidden_keras(data, self.layer_index)

