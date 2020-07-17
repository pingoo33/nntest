import random
from collections import defaultdict

from model.interface.model_manager import ModelManager
from model.threshold_manager import ThresholdManager
from test.interface.FCL_coverage import FCLCoverage
import numpy as np
import matplotlib
matplotlib.use('agg')
from matplotlib import pyplot as plt


class KMultisectionCoverage(FCLCoverage):
    def __init__(self, layer, model_manager: ModelManager, threshold_manager: ThresholdManager, size):

        self.num_section = size
        self.plt_x = []
        self.plt_y = []
        self.fr_plt_x = []
        self.fr_plt_y = [[]] * size

        self.layer = layer
        self.model_manager = model_manager
        self.threshold_manager = threshold_manager

        self.covered_dicts = [defaultdict(bool)] * size
        self.__init_covered_dict()
        self.frequency_dicts = [defaultdict(int)] * size
        self.__init_frequency_dict()

    def __init_covered_dict(self):
        for dict in self.covered_dicts:
            for index in range(self.layer.output_shape[-1]):
                dict[index] = False

    def __init_frequency_dict(self):
        for dict in self.frequency_dicts:
            for index in range(self.layer.output_shape[-1]):
                dict[index] = 0

    @staticmethod
    def calculate_variation(datas):
        sum_y = 0
        for data in datas:
            for d in data:
                y = d
                sum_y += y
        mean = sum_y / len(datas[0])

        square_sum = 0
        for data in datas:
            for d in data:
                y = d
                square_sum += (y - mean) ** 2
        variation = square_sum / len(datas[0])

        return mean, variation

    def calculate_coverage(self):
        size = self.layer.output_shape[-1]
        total_number_neurons = size * self.num_section
        covered_number_neurons = 0
        for dict in self.covered_dicts:
            for index in range(size):
                if dict[index] is True:
                    covered_number_neurons += 1

        return covered_number_neurons, covered_number_neurons / float(total_number_neurons)

    def update_features(self, data):
        inter_output = self.model_manager.get_intermediate_output(self.layer, data)
        for num_neuron in range(inter_output.shape[-1]):
            max_threshold = self.threshold_manager.get_max_threshold(num_neuron)
            min_threshold = self.threshold_manager.get_min_threshold(num_neuron)
            delta = (max_threshold - min_threshold) / self.num_section

            index = 0
            threshold = min_threshold
            while index < self.num_section:
                if threshold <= np.mean(inter_output[..., num_neuron]) < threshold + delta:
                    self.covered_dicts[index][num_neuron] = True
                    self.frequency_dicts[index][num_neuron] += 1
                threshold += delta
                index += 1

    def update_graph(self, num_samples):
        _, coverage = self.calculate_coverage()
        self.plt_x.append(num_samples)
        self.plt_y.append(coverage)
        print("%s layer k-multisection coverage : %.8f" % (self.layer.name, coverage))

    def update_frequency_graph(self):
        for num_neuron in range(self.layer.output_shape[-1]):
            self.fr_plt_x.append(num_neuron)

        for index in range(self.num_section):
            temp = []
            for num_neuron in range(self.layer.output_shape[-1]):
                temp.append(self.frequency_dicts[index][num_neuron])
            self.fr_plt_y[index] = temp

    def display_graph(self):
        plt.plot(self.plt_x, self.plt_y)
        plt.xlabel('# of generated samples')
        plt.ylabel('coverage')
        plt.title('K-Multisection Coverage of ' + self.layer.name)
        plt.savefig('output/' + self.model_manager.model_name + '/' + self.layer.name + '_kc.png')
        plt.clf()

    def display_frequency_graph(self):
        n_groups = len(self.fr_plt_x)
        index = np.arange(n_groups)

        plt.bar(index, self.fr_plt_y[0], align='center')
        for i in range(1, self.num_section):
            r = random.random()
            g = random.random()
            b = random.random()
            a = random.random()
            plt.bar(index, self.fr_plt_y[i], align='center', color=(r, g, b, a),
                    bottom=self.fr_plt_y[i - 1])

        plt.xlabel('features')
        plt.ylabel('number of activation')
        plt.title(self.layer.name + ' Frequency')
        plt.xlim(-1, n_groups)
        plt.savefig('output/' + self.model_manager.model_name + '/' + self.layer.name + '_kc_Frequency.png')
        plt.clf()

    def display_stat(self):
        mean, variation = self.calculate_variation(self.fr_plt_y)

        f = open('output/%s_%s_tc.txt' % (self.model_manager.model_name, self.layer.name), 'w')
        f.write('mean: %f' % mean)
        f.write('variation: %f' % variation)
        f.close()
