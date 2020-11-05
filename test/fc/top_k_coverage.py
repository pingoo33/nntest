import numpy as np
import pandas as pd
import matplotlib

matplotlib.use('agg')
from matplotlib import pyplot as plt
from collections import defaultdict

from model.interface.model_manager import ModelManager
from test.interface.FCL_coverage import FCLCoverage


class TopKCoverage(FCLCoverage):
    def save_feature(self):
        pass

    def __init__(self, layer, model_manager: ModelManager, size):
        if layer.output_shape[-1] < size:
            self.size = layer.output_shape[-1]
        else:
            self.size = size

        self.name = "TopKCoverage"
        self.layer = layer
        self.model_manager = model_manager

        self.plt_x = []
        self.plt_y = []
        self.fr_plt_x = []
        self.fr_plt_y = []

        self.covered_dict = defaultdict(bool)
        self.__init_covered_dict()
        self.frequency_dict = defaultdict(int)
        self.__init_frequency_dict()

    def __init_covered_dict(self):
        for index in range(self.size):
            self.covered_dict[index] = False

    def __init_frequency_dict(self):
        for index in range(self.layer.output_shape[-1]):
            self.frequency_dict[index] = 0

    def calculate_coverage(self):
        total_number_neurons = self.layer.output_shape[-1]
        covered_number_neurons = 0
        for index in range(total_number_neurons):
            if self.covered_dict[index] is True:
                covered_number_neurons += 1

        return covered_number_neurons, covered_number_neurons / float(total_number_neurons)

    def update_features(self, data):
        inter_output = self.model_manager.get_intermediate_output(self.layer, data)
        neuron_outputs = []
        for num_neuron in range(inter_output.shape[-1]):
            neuron_outputs.append(np.mean(inter_output[..., num_neuron]))

        neuron_outputs.sort(reverse=True)
        for index in range(self.size):
            for num_neuron in range(inter_output.shape[-1]):
                if np.mean(inter_output[..., num_neuron]) == neuron_outputs[index]:
                    self.covered_dict[num_neuron] = True
                    self.frequency_dict[num_neuron] += 1

    def update_graph(self, num_samples):
        _, coverage = self.calculate_coverage()
        self.plt_x.append(num_samples)
        self.plt_y.append(coverage)
        # print("%s layer Top-k coverage : %.8f" % (self.layer.name, coverage))

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
        for num_neuron in range(self.layer.output_shape[-1]):
            self.fr_plt_x.append(num_neuron)
            self.fr_plt_y.append(self.frequency_dict[num_neuron])

    def display_graph(self, name=''):
        name = name + self.name
        plt.plot(self.plt_x, self.plt_y)
        plt.xlabel('# of generated samples')
        plt.ylabel('coverage')
        plt.title('Top-k Coverage of ' + self.layer.name)
        plt.savefig('output/' + self.model_manager.model_name + '/' + self.layer.name + '_' + name + '.png')
        plt.clf()

    def display_frequency_graph(self, name=''):
        name = name + self.name
        self.fr_plt_y = np.array(self.fr_plt_y)
        df = pd.DataFrame(self.fr_plt_y)

        title = self.layer.name + ' Frequency of Top-K Coverage'
        ax = df.plot(kind='bar', figsize=(10, 6), title=title,
                     xticks=([w for w in range(len(self.fr_plt_x)) if w % 10 == 0]))
        ax.set_xlabel('neuron')
        ax.set_ylabel('number of activation')
        plt.savefig('output/' + self.model_manager.model_name + '/' + self.layer.name + '_' + name + '_Frequency.png')
        plt.clf()

    def display_stat(self, name=''):
        _, coverage = self.calculate_coverage()
        mean, variation = self.calculate_variation(self.fr_plt_y)

        f = open('output/%s/%s_%s_tkc.txt' % (self.model_manager.model_name, name, self.layer.name), 'w')
        f.write('coverage: %f\n' % coverage)
        f.write('mean: %f\n' % mean)
        f.write('variation: %f' % variation)
        f.close()

    def get_name(self):
        return self.name
