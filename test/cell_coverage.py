from collections import defaultdict

from model.interface.model_manager import ModelManager
from model.state_manager import StateManager
from test.interface.RL_coverage import RLCoverage
import numpy as np
from matplotlib import pyplot as plt


class CellCoverage(RLCoverage):
    def __init__(self, layer, model_manager: ModelManager, state_manager: StateManager, threshold, data):
        self.plt_x = []
        self.plt_y = []
        self.fr_plt_x = []
        self.fr_plt_y = []

        self.layer = layer
        self.model_manager = model_manager
        self.state_manager = state_manager
        self.threshold = threshold
        self.hidden = self.state_manager.get_hidden_state(data)
        activation = self.__get_activation()
        self.total_feature = len((np.argwhere(activation >= np.min(activation))).tolist())

        self.covered_dict = defaultdict(bool)
        self.frequency_dict = defaultdict(int)
        self.__init_covered_dict()
        self.__init_frequency_dict()

    def __init_covered_dict(self):
        for index in range(self.total_feature):
            self.covered_dict[index] = False

    def __init_frequency_dict(self):
        for index in range(self.total_feature):
            self.frequency_dict[index] = 0

    def __get_activation(self):
        hidden = self.hidden
        alpha1 = np.sum(np.where(hidden > 0, hidden, 0), axis=1)
        alpha2 = np.sum(np.where(hidden < 0, hidden, 0), axis=1)
        alpha11 = np.insert(np.delete(alpha1, -1, axis=0), 0, 0, axis=0)
        alpha22 = np.insert(np.delete(alpha2, -1, axis=0), 0, 0, axis=0)
        alpha = np.abs(alpha1 - alpha11) + np.abs(alpha2 - alpha22)
        return alpha

    def calculate_coverage(self):
        covered_number_neurons = 0
        for index in range(self.total_feature):
            if self.covered_dict[index] is True:
                covered_number_neurons += 1

        return covered_number_neurons, covered_number_neurons / float(self.total_feature)

    def update_features(self, data):
        self.hidden = self.state_manager.get_hidden_state(data)
        activation = self.__get_activation()
        features = (np.argwhere(activation > self.threshold)).tolist()
        for feature in features:
            self.covered_dict[feature[0]] = True
            self.frequency_dict[feature[0]] += 1

    def update_graph(self, num_samples):
        _, coverage = self.calculate_coverage()
        self.plt_x.append(num_samples)
        self.plt_y.append(coverage)
        print("%s layer cell coverage : %.8f" % (self.layer.name, coverage))

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
        for index in range(self.total_feature):
            self.fr_plt_x.append(index)
            self.fr_plt_y.append(self.frequency_dict[index])

    def display_graph(self):
        plt.plot(self.plt_x, self.plt_y)
        plt.xlabel('# of generated samples')
        plt.ylabel('coverage')
        plt.title('Cell Coverage of ' + self.layer.name)
        plt.savefig('output/' + self.model_manager.model_name + '/' + self.layer.name + '_cc.png')
        plt.clf()

    def display_frequency_graph(self):
        n_groups = len(self.fr_plt_x)
        index = np.arange(n_groups)

        plt.bar(index, self.fr_plt_y, align='center')

        plt.xlabel('features')
        plt.ylabel('number of activation')
        plt.title(self.layer.name + ' Frequency')
        plt.xlim(-1, n_groups)
        plt.savefig('output/' + self.model_manager.model_name + '/' + self.layer.name + '_cc_Frequency.png')
        plt.clf()