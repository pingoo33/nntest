from model.state_manager import StateManager
from test.interface.RL_coverage import RLCoverage
import numpy as np
import matplotlib
matplotlib.use('agg')
from matplotlib import pyplot as plt
from collections import defaultdict
import pandas as pd

from model.state_manager import StateManager
from test.interface.RL_coverage import RLCoverage


class GateCoverage(RLCoverage):

    def save_feature(self):
        activations = pd.DataFrame(self.activates)
        activations.to_csv('gc_activates.csv', mode='w')

    def __init__(self, layer, model_name, state_manager: StateManager, threshold, data):
        self.name = "GateCoverage"
        self.plt_x = []
        self.plt_y = []
        self.fr_plt_x = []
        self.fr_plt_y = []

        self.layer = layer
        self.model_name = model_name
        self.state_manager = state_manager
        self.threshold = threshold
        self.gate = self.state_manager.get_forget_state(data)
        activation = self.get_activation()
        self.total_feature = len((np.argwhere(activation >= np.min(activation))).tolist())

        self.covered_dict = defaultdict(bool)
        self.frequency_dict = defaultdict(int)
        self.__init_covered_dict()
        self.__init_frequency_dict()
        self.activates = []

    def __init_covered_dict(self):
        for index in range(self.total_feature):
            self.covered_dict[index] = False

    def __init_frequency_dict(self):
        for index in range(self.total_feature):
            self.frequency_dict[index] = 0

    def get_activation(self):
        gate = self.gate
        alpha = np.sum(gate, axis=1) / float(gate.shape[1])
        # alpha = np.sum(np.where(gate > 0.8, 1, 0), axis=1)/float(gate.shape[1])
        return alpha

    def calculate_coverage(self):
        covered_number_neurons = 0
        for index in range(self.total_feature):
            if self.covered_dict[index] is True:
                covered_number_neurons += 1

        return covered_number_neurons, covered_number_neurons / float(self.total_feature)

    def update_features(self, data):
        self.gate = self.state_manager.get_forget_state(data)
        activation = self.get_activation()
        self.activates.append(activation)
        features = (np.argwhere(activation > self.threshold)).tolist()
        for feature in features:
            self.covered_dict[feature[0]] = True
            self.frequency_dict[feature[0]] += 1

    def update_graph(self, num_samples):
        _, coverage = self.calculate_coverage()
        self.plt_x.append(num_samples)
        self.plt_y.append(coverage)
        # print("%s layer gate coverage : %.8f" % (self.layer.name, coverage))

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
        plt.title('Gate Coverage of ' + self.layer.name)
        plt.savefig('output/' + self.model_name + '/' + self.layer.name + '_gc.png')
        plt.clf()

    def display_frequency_graph(self):
        self.fr_plt_y = np.array(self.fr_plt_y)
        df = pd.DataFrame(self.fr_plt_y)

        title = self.layer.name + ' Frequency of Gate Coverage'
        ax = df.plot(kind='bar', figsize=(10, 6), title=title,
                     xticks=([w for w in range(len(self.fr_plt_x)) if w % 10 == 0]))
        ax.set_xlabel('state')
        ax.set_ylabel('number of activation')
        plt.savefig('output/' + self.model_name + '/' + self.layer.name + '_cc_Frequency.png')
        plt.clf()

    def display_stat(self):
        _, coverage = self.calculate_coverage()
        mean, variation = self.calculate_variation(self.fr_plt_y)

        f = open('output/%s_%s_gc.txt' % (self.model_name, self.layer.name), 'w')
        f.write('coverage: %f\n' % coverage)
        f.write('mean: %f\n' % mean)
        f.write('variation: %f' % variation)
        f.close()

    def get_name(self):
        return self.name
