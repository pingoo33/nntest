from collections import defaultdict
from model.state_manager import StateManager
from test.interface.RL_coverage import RLCoverage
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use('agg')
from matplotlib import pyplot as plt


class BoundaryCoverage(RLCoverage):
    def __init__(self, layer, model_name, state_manager: StateManager, avg, min, max, data):
        self.name = "BoundaryCoverage"

        # Thresholds
        self.avg = avg
        self.min = min
        self.max = max

        self.plt_x = []
        self.plt_y = []
        self.fr_plt_x = []
        self.fr_plt_y = []

        self.layer = layer
        self.model_name = model_name
        self.state_manager = state_manager
        self.hidden = None

        self.feature_length = 0
        self.__init_feature_length([data])

        self.covered_dict = defaultdict(bool)
        self.frequency_dict = defaultdict(int)
        self.__init_covered_dict()
        self.__init_frequency_dict()
        self.activates = []

    def __init_feature_length(self, data):
        self.hidden = self.state_manager.get_hidden_state(data)
        activation = self.get_activation()
        self.feature_length = len((np.argwhere(activation >= np.min(activation))).tolist())

    def __init_covered_dict(self):
        for index in range(self.feature_length):
            self.covered_dict[index] = False

    def __init_frequency_dict(self):
        for index in range(self.feature_length):
            self.frequency_dict[index] = 0

    def get_activation(self):
        alpha1 = np.sum(np.where(self.hidden > 0, self.hidden, 0), axis=2)
        alpha2 = np.sum(np.where(self.hidden < 0, self.hidden, 0), axis=2)
        alpha = np.abs(alpha1 + alpha2)
        return alpha

    def calculate_coverage(self):
        covered_number_neurons = 0
        for index in range(self.feature_length):
            if self.covered_dict[index] is True:
                covered_number_neurons += 1

        return covered_number_neurons, covered_number_neurons / float(self.feature_length)

    def update_features(self, data):
        self.hidden = self.state_manager.get_hidden_state([data])
        activation = self.get_activation()
        activation = (activation - self.min) / (self.max - self.min)
        print(activation)
        print(self.avg)
        fitness = np.min((self.avg - activation), axis=0)

        features = (np.argwhere(fitness <= 0)).tolist()

        for feature in features:
            self.covered_dict[feature[0]] = True
            self.frequency_dict[feature[0]] += 1

    def update_graph(self, num_samples):
        _, coverage = self.calculate_coverage()
        print(coverage)
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
        for index in range(self.feature_length):
            self.fr_plt_x.append(index)
            self.fr_plt_y.append(self.frequency_dict[index])

    def display_graph(self):
        plt.plot(self.plt_x, self.plt_y)
        plt.xlabel('# of generated samples')
        plt.ylabel('coverage')
        plt.title('Boundary Coverage of ' + self.layer.name)
        plt.savefig('output/' + self.model_name + '/' + self.layer.name + '_bc.png')
        plt.clf()

    def display_frequency_graph(self):
        self.fr_plt_y = np.array(self.fr_plt_y)
        df = pd.DataFrame(self.fr_plt_y)

        title = self.layer.name + ' Frequency of Boundary Coverage'
        ax = df.plot(kind='bar', figsize=(10, 6), title=title,
                     xticks=([w for w in range(len(self.fr_plt_x)) if w % 10 == 0]))
        ax.set_xlabel('state')
        ax.set_ylabel('number of activation')
        plt.savefig('output/' + self.model_name + '/' + self.layer.name + '_bc_Frequency.png')
        plt.clf()

    def display_stat(self):
        _, coverage = self.calculate_coverage()
        mean, variation = self.calculate_variation(self.fr_plt_y)

        f = open('output/%s_%s_cc.txt' % (self.model_name, self.layer.name), 'w')
        f.write('coverage: %f\n' % coverage)
        f.write('mean: %f\n' % mean)
        f.write('variation: %f' % variation)
        f.close()

    def get_name(self):
        return self.name

    def save_feature(self):
        pass
