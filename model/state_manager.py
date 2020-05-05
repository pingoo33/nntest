import numpy as np
import keras.backend as K


class StateManager:
    def __init__(self, model, layer_index):
        self.model = model
        self.layer_index = layer_index

    def __evaluate(self, nodes_to_evaluate, x, y=None):
        symb_inputs = (self.model._feed_inputs + self.model._feed_targets
                       + self.model._feed_sample_weights)
        f = K.function(symb_inputs, nodes_to_evaluate)
        x_, y_, sample_weight_ = self.model._standardize_user_data(x, y)
        return f(x_ + y_ + sample_weight_)

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
        i = 0
        y = np.zeros((1, len(x[0, :])))
        for x_i in x[0, :]:
            if x_i < -2.5:
                y_i = 0
            elif x_i > 2.5:
                y_i = 1
            else:
                y_i = 0.2 * x_i + 0.5
            y[0, i] = y_i
            i = i + 1
        return y

    # calculate the lstm hidden state and cell state manually (no dropout)
    # activation function is tanh
    def cal_hidden_state(self, test, layer_num, data_size=12):
        if layer_num == 0:
            acx = test
        else:
            acx = self.__get_activations_single_layer(np.array([test]),
                                                      self.model.layers[layer_num - 1].name)

        units = int(int(self.model.layers[layer_num].trainable_weights[0].shape[1]) / 4)
        # lstm_layer = model.layers[1]
        w = self.model.layers[layer_num].get_weights()[0]
        u = self.model.layers[layer_num].get_weights()[1]
        b = self.model.layers[layer_num].get_weights()[2]

        w_i = w[:, :units]
        w_f = w[:, units: units * 2]
        w_c = w[:, units * 2: units * 3]
        w_o = w[:, units * 3:]

        u_i = u[:, :units]
        u_f = u[:, units: units * 2]
        u_c = u[:, units * 2: units * 3]
        u_o = u[:, units * 3:]

        b_i = b[:units]
        b_f = b[units: units * 2]
        b_c = b[units * 2: units * 3]
        b_o = b[units * 3:]

        # calculate the hidden state value
        h_t = np.zeros((data_size, units))
        c_t = np.zeros((data_size, units))
        f_t = np.zeros((data_size, units))
        h_t0 = np.zeros((1, units))
        c_t0 = np.zeros((1, units))

        for i in range(0, data_size):
            f_gate = self.__hard_sigmoid(np.dot(acx[i, :], w_f) + np.dot(h_t0, u_f) + b_f)
            i_gate = self.__hard_sigmoid(np.dot(acx[i, :], w_i) + np.dot(h_t0, u_i) + b_i)
            o_gate = self.__hard_sigmoid(np.dot(acx[i, :], w_o) + np.dot(h_t0, u_o) + b_o)
            new_c = np.tanh(np.dot(acx[i, :], w_c) + np.dot(h_t0, u_c) + b_c)
            c_t0 = f_gate * c_t0 + i_gate * new_c
            h_t0 = o_gate * np.tanh(c_t0)
            c_t[i, :] = c_t0
            h_t[i, :] = h_t0
            f_t[i, :] = f_gate

        return h_t, c_t, f_t

    def get_hidden_state(self, data):
        hidden, _, _ = self.cal_hidden_state(data, self.layer_index)
        return hidden

    def get_forget_state(self, data):
        _, _, gate = self.cal_hidden_state(data, self.layer_index)
        return gate
