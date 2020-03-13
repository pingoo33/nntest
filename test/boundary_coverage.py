from collections import defaultdict

from model.interface.model_manager import ModelManager
from model.threshold_manager import ThresholdManager
from test.interface.coverage import Coverage
import numpy as np
from matplotlib import pyplot as plt


class BoundaryCoverage(Coverage):
    def __init__(self, layer, model_manager: ModelManager, threshold_manager: ThresholdManager):
        if layer.name == threshold_manager.layer.name:
            sys.exit("Unmatch layer: BoundaryCoverage")

        self.plt_x = []
        self.plt_y = []
        self.fr_plt_x = []
        self.max_fr_plt_y = []
        self.min_fr_plt_y = []

        self.layer = layer
        self.model_manager = model_manager
        self.threshold_manager = threshold_manager

        self.max_covered_dict = defaultdict(bool)
        self.min_covered_dict = defaultdict(bool)
        self.__init_covered_dict()
        self.max_frequency_dict = defaultdict(int)
        self.min_frequency_dict = defaultdict(int)
        self.__init_frequency_dict()

    def __init_covered_dict(self):
        for index in range(self.layer.output_shape[-1]):
            self.max_covered_dict[index] = False
            self.min_covered_dict[index] = False

    def __init_frequency_dict(self):
        for index in range(self.layer.output_shape[-1]):
            self.max_frequency_dict[index] = 0
            self.min_frequency_dict[index] = 0

    def calculate_coverage(self):
        size = self.layer.output_shape[-1]
        total_number_neurons = size * 2
        covered_number_neurons = 0
        for index in range(size):
            if self.max_covered_dict[index] is True:
                covered_number_neurons += 1
            if self.min_covered_dict[index] is True:
                covered_number_neurons += 1

        return covered_number_neurons, covered_number_neurons / float(total_number_neurons)

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

    def update_features(self, data):
        inter_output = self.model_manager.get_intermediate_output(self.layer, data)
        for num_neuron in range(inter_output.shape[-1]):
            max_threshold = self.threshold_manager.get_max_threshold(num_neuron)
            min_threshold = self.threshold_manager.get_min_threshold(num_neuron)

            if np.mean(inter_output[..., num_neuron]) > max_threshold:
                self.max_covered_dict[num_neuron] = True
                self.max_frequency_dict[num_neuron] += 1
            if np.mean(inter_output[..., num_neuron]) < min_threshold:
                self.min_covered_dict[num_neuron] = True
                self.min_frequency_dict[num_neuron] += 1

    def update_graph(self, num_samples):
        _, coverage = self.calculate_coverage()
        self.plt_x.append(num_samples)
        self.plt_y.append(coverage)
        print("%s layer boundary coverage : %.8f" % (self.layer.name, coverage))

    def update_frequency_graph(self):
        for num_neuron in range(self.layer.output_shape[-1]):
            self.fr_plt_x.append(num_neuron)
            self.max_fr_plt_y.append(self.max_frequency_dict[num_neuron])
            self.min_fr_plt_y.append(self.min_frequency_dict[num_neuron])

    def display_graph(self):
        plt.plot(self.plt_x, self.plt_y)
        plt.xlabel('# of generated samples')
        plt.ylabel('coverage')
        plt.title('Boundary Coverage of ' + self.layer.name)
        plt.savefig('output/' + self.model_manager.model_name + '/' + self.layer.name + '_bc.png')
        plt.clf()

    def display_frequency_graph(self):
        n_groups = len(self.fr_plt_x)
        index = np.arrange(n_groups)

        plt.bar(index, self.max_frequency_dict, align='center')
        plt.bar(index, self.min_frequency_dict, align='center', color='#5233aa', bottom=self.max_frequency_dict)

        plt.xlabel('features')
        plt.ylabel('activate counts')
        plt.title(self.layer.name + ' Frequency')
        plt.xlim(-1, n_groups)
        plt.savefig('output/' + self.model_manager.model_name + '/' + self.layer.name + '_bc_Frequency.png')
        plt.clf()
