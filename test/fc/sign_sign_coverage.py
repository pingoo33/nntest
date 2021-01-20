import numpy as np
from collections import defaultdict
import pandas as pd
import matplotlib

matplotlib.use('agg')
from matplotlib import pyplot as plt

from model.interface.model_manager import ModelManager
from test.interface.FCL_coverage import FCLCoverage


class SignSignCoverage(FCLCoverage):
    def __init__(self, first_layer, second_layer, model_manager: ModelManager):
        self.name = "Sign-SignCoverage"
        self.plt_x = []
        self.plt_y = []
        self.fr_plt_x = []
        self.fr_plt_y = []

        self.first = first_layer
        self.second = second_layer
        self.model_manager = model_manager

        self.total_num_features = self.first.output_shape[-1] * self.second.output_shape[-1]

        self.sign_change_dict = [defaultdict(bool)] * 2
        self.covered_dict = defaultdict(bool)
        self.__init_covered_dict()
        self.frequency_dict = defaultdict(int)
        self.__init_frequency_dict()

    def __init_sign_change_dict(self, datas):
        self.sign_change_dict = [defaultdict(bool)] * 2
        layer_output_l1_d1 = self.model_manager.get_intermediate_output(self.first, datas[0])
        layer_output_l1_d2 = self.model_manager.get_intermediate_output(self.first, datas[1])
        layer_output_l2_d1 = self.model_manager.get_intermediate_output(self.second, datas[0])
        layer_output_l2_d2 = self.model_manager.get_intermediate_output(self.second, datas[1])

        self.__update_sign_change_dict(
            [layer_output_l1_d1, layer_output_l1_d2], layer_index=0
        )
        self.__update_sign_change_dict(
            [layer_output_l2_d1, layer_output_l2_d2], layer_index=1
        )

    def __update_sign_change_dict(self, layer_outputs, layer_index):
        first_data_output = layer_outputs[0]
        second_data_output = layer_outputs[1]

        for num_neuron in range(first_data_output.shape[-1]):
            f_output = np.mean(first_data_output[..., num_neuron])
            s_output = np.mean(second_data_output[..., num_neuron])

            if (f_output * s_output) > 0 or (f_output == 0 and s_output == 0):
                self.sign_change_dict[layer_index][num_neuron] = False
            else:
                self.sign_change_dict[layer_index][num_neuron] = True

    def __init_covered_dict(self):
        for index in range(self.total_num_features):
            self.covered_dict[index] = False

    def __init_frequency_dict(self):
        for index in range(self.total_num_features):
            self.frequency_dict[index] = 0

    def calculate_coverage(self):
        covered_number_features = 0
        for index in range(self.total_num_features):
            if self.covered_dict[index] is True:
                covered_number_features += 1

        return covered_number_features, covered_number_features / float(self.total_num_features)

    def update_features(self, data):
        self.__init_sign_change_dict(data)

        f_size = self.first.output_shape[-1]
        s_size = self.second.output_shape[-1]

        for f_index in range(f_size):
            if self.sign_change_dict[0][f_index] is False:
                continue
            nsc_flag = True
            for check in range(f_size):
                if check == f_index:
                    continue
                if self.sign_change_dict[0][check] is True:
                    nsc_flag = False
            if nsc_flag is False:
                break
            for s_index in range(s_size):
                if self.sign_change_dict[1][s_index] is True:
                    self.covered_dict[f_index * f_size + s_index] = True
                    self.frequency_dict[f_index * f_size + s_index] += 1

    def update_graph(self, num_samples):
        _, coverage = self.calculate_coverage()
        self.plt_x.append(num_samples)
        self.plt_y.append(coverage)

    @staticmethod
    def calculate_variation(data):
        sum_y = 0

        for y in data:
            sum_y += y

        mean = sum_y / len(data)

        square_sum = 0
        for y in data:
            square_sum += (y - mean) ** 2

        variation = square_sum / len(data)
        return mean, variation

    def update_frequency_graph(self):
        for index in range(self.total_num_features):
            self.fr_plt_x.append(index)
            self.fr_plt_y.append(self.frequency_dict[index])

    def display_graph(self, name=''):
        name = name + self.name
        plt.plot(self.plt_x, self.plt_y)
        plt.xlabel('# of generated samples')
        plt.ylabel('coverage')
        plt.title('Sign-Sign Coverage of ' + self.first.name)
        plt.savefig('output/' + self.model_manager.model_name + '/' + self.first.name + '_' + name + '.png')
        plt.clf()

    def display_frequency_graph(self, name=''):
        name = name + self.name
        self.fr_plt_y = np.array(self.fr_plt_y)
        df = pd.DataFrame(self.fr_plt_y)

        title = self.first.name + ' Frequency of Sign-Sign Coverage'
        ax = df.plot(kind='bar', figsize=(10, 6), title=title,
                     xticks=([w for w in range(len(self.fr_plt_x)) if w % 10 == 0]))
        ax.set_xlabel('index')
        ax.set_ylabel('number of activation')
        plt.savefig('output/' + self.model_manager.model_name + '/' + self.first.name + '_' + name + '_Frequency.png')
        plt.clf()

    def display_stat(self, name):
        _, coverage = self.calculate_coverage()
        mean, variation = self.calculate_variation(self.fr_plt_y)
        
        f = open('output/%s/%s_%s_ssc.txt' % (self.model_manager.model_name, name, self.first.name), 'w')
        f.write('coverage: %f\n' % coverage)
        f.write('mean: %f\n' % mean)
        f.write('variation: %f' % variation)
        f.close()

    def get_name(self):
        return self.name

    def save_feature(self):
        pass
