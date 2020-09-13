import math
import numpy as np
import matplotlib

matplotlib.use('agg')
from matplotlib import pyplot as plt

from model.interface.model_manager import ModelManager
from test.interface.FCL_coverage import FCLCoverage


class TopKPatternCoverage(FCLCoverage):
    def __init__(self, model_manager: ModelManager, size):
        self.model_manager = model_manager
        _, self.layers = model_manager.get_fc_layer()
        if self.layers[-1].output_shape[-1] < size:
            self.size = self.layers[-1].output_shape[-1]
        else:
            self.size = size

        self.name = "TopKPatternCoverage"
        self.plt_x = []
        self.plt_y = []

        self.num_patterns = self.__init_pattern_number()
        self.covered_pattern = []

    def __init_pattern_number(self):
        num_neurons_by_layer = []
        for layer in self.layers:
            num_neurons_by_layer.append(layer.output_shape[-1])

        num_patterns = 1
        for num_neurons in num_neurons_by_layer:
            num_patterns = num_patterns * (math.factorial(num_neurons) / (
                    math.factorial(num_neurons - self.size) * math.factorial(self.size)))

        return num_patterns

    def calculate_coverage(self):
        return len(self.covered_pattern), len(self.covered_pattern) / self.num_patterns

    def update_features(self, data):
        pattern = []
        for layer in self.layers:
            inter_output = self.model_manager.get_intermediate_output(layer, data)
            neuron_outputs = []
            for num_neuron in range(inter_output.shape[-1]):
                neuron_outputs.append(np.mean(inter_output[..., num_neuron]))

            pattern.append(self.__get_top_k_neuron_id(neuron_outputs))

        if not (pattern in self.covered_pattern):
            self.covered_pattern.append(pattern)

    def __get_top_k_neuron_id(self, neuron_outputs):
        neuron_outputs.sort(reverse=True)

        top_k_indices = []
        for index_sorted in range(self.size):
            for index, output in enumerate(neuron_outputs):
                if neuron_outputs[index_sorted] == output:
                    top_k_indices.append(index)

        return top_k_indices.sort()

    def update_graph(self, num_samples):
        _, coverage = self.calculate_coverage()
        self.plt_x.append(num_samples)
        self.plt_y.append(coverage)
        # print("Top-k pattern coverage : %.8f" % coverage)

    @staticmethod
    def calculate_variation(data):
        pass

    def update_frequency_graph(self):
        pass

    def display_graph(self):
        plt.plot(self.plt_x, self.plt_y)
        plt.xlabel('# of generated samples')
        plt.ylabel('coverage')
        plt.title('Top-k Pattern Coverage of ' + self.model_manager.model_name)
        plt.savefig('output/' + self.model_manager.model_name + '_tkpc.png')
        plt.clf()

    def display_frequency_graph(self):
        pass

    def display_stat(self):
        pass

    def get_name(self):
        return self.name
