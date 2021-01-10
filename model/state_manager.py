import numpy as np
from tensorflow.keras.models import Model
import keras.backend as K
from tensorflow.keras.layers import Input, LSTM


class StateManager:
    def __init__(self, model: Model, layer_index):
        self.model = model
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

    def __evaluate(self, nodes_to_evaluate, x, y=None):
        symb_inputs = self.model._feed_inputs
        f = K.function(symb_inputs, nodes_to_evaluate)
        x_ = [np.array(x)]
        if y is None:
            y = []
        return f(x_ + y)

    def __get_activations_single_layer(self, x, layer_name=None):
        nodes = [layer.output for layer in self.model.layers if
                 layer.name == layer_name or layer_name is None]
        # we process the placeholders later (Inputs node in Keras). Because there's a bug in Tensorflow.
        input_layer_outputs, layer_outputs = [], []
        [input_layer_outputs.append(node) if 'input_' in node.name else layer_outputs.append(node) for node in nodes]
        activations = self.__evaluate(layer_outputs, x)
        activations_dict = dict(zip([output.name for output in layer_outputs], activations))
        activations_inputs_dict = dict(zip([output.name for output in input_layer_outputs], x))
        result = activations_inputs_dict.copy()
        result.update(activations_dict)
        return np.squeeze(list(result.values())[0])

    @staticmethod
    def __hard_sigmoid(x):
        return np.maximum(0, np.minimum(1, 0.2 * x + 0.5))

    def cal_hidden_keras(self, test, layer_num):
        if layer_num == 0:
            acx = test
        else:
            acx = self.__get_activations_single_layer(np.array(test), self.model.layers[layer_num - 1].name)

        units = int(int(self.model.layers[layer_num].trainable_weights[0].shape[1]) / 4)

        if len(np.shape(acx)) < len(np.shape(test)):
            acx = np.array([acx])

        inp = Input(batch_shape=(None, acx.shape[1], acx.shape[2]), name="input")
        rnn, s, c = LSTM(units,
                         return_sequences=True,
                         stateful=False,
                         return_state=True,
                         name="RNN")(inp)
        states = Model(inputs=[inp], outputs=[s, c, rnn])

        for layer in states.layers:
            if layer.name == "RNN":
                layer.set_weights(self.model.layers[layer_num].get_weights())

        h_t, c_t, rnn = states.predict(acx)

        return rnn

    def get_hidden_state(self, data):
        return self.cal_hidden_keras(data, self.layer_index)
