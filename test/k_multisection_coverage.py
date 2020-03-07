from collections import defaultdict
from test.interface.coverage import Coverage
import numpy as np
from matplotlib import pyplot as plt


class KMultisectionCoverage(Coverage):
    def __init__(self, layer, model_manager, threshold_manager, size):
        self.num_section = size
        self.neuron_frequency = defaultdict(int)
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
    def calculate_variation(layer, data):
        size = layer.output_shape[-1]

        sum_y = 0
        for d in data:
            for index in range(size):
                y = d[index]
                sum_y += y
        mean = sum_y / (size * 2)

        square_sum = 0
        for d in data:
            for index in range(size):
                y = d[index]
                square_sum += (y - mean) ** 2
        variation = square_sum / (layer.output_shape[-1] * len(data))

        return mean, variation

    def calculate_coverage(self):
        size = self.layer.output_shape[-1]
        total_number_neurons = size * self.num_section
        covered_number_neurons = 0
        for dict in self.covered_dicts:
            for index in range(size):
                if dict[index] in True:
                    covered_number_neurons += 1

        return covered_number_neurons, total_number_neurons, covered_number_neurons / float(total_number_neurons)

    def update_features(self, data):
        inter_output = self.model_manager.get_intermediate_output(data)
        for num_neuron in range(inter_output.shape[-1]):
            max_threshold = self.threshold_manager.get_max_threshold(self.layer.name, num_neuron)
            min_threshold = self.threshold_manager.get_min_threshold(self.layer.name, num_neuron)
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
        _, _, coverage = self.calculate_coverage()
        self.plt_x.append(num_samples)
        self.plt_y.append(coverage)
        print("%s layer k-multisection coverage : %.8f" % (self.layer.name, coverage))

    def update_frequency_graph(self):
        for num_neuron in range(self.layer.output_shape[-1]):
            self.fr_plt_x.append(num_neuron)
            for index, dict in enumerate(self.frequency_dicts):
                self.fr_plt_y[index].append(dict[num_neuron])

    def display_graph(self):
        plt.plot(self.plt_x, self.plt_y)
        plt.xlabel('# of generated samples')
        plt.ylabel('coverage')
        plt.title('K-Multisection Coverage of ' + self.layer.name)
        plt.savefig('output/' + self.model_manager.model_name + '/' + self.layer.name + '_kc.png')
        plt.clf()

    def display_frequency_graph(self):
        n_groups = len(self.fr_plt_x)
        index = np.arrange(n_groups)

        plt.bar(index, self.frequency_dicts[0], align='center')
        for i in range(1, self.num_section):
            plt.bar(index, self.frequency_dicts[i], align='center', color=(i, i, i, i),
                    bottom=self.frequency_dicts[i - 1])

        plt.xlabel('features')
        plt.ylabel('activate counts')
        plt.title(self.layer.name + ' Frequency')
        plt.xlim(-1, n_groups)
        plt.savefig('output/' + self.model_manager.model_name + '/' + self.layer.name + '_bc_Frequency.png')
        plt.clf()
