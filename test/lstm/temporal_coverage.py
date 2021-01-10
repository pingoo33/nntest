from collections import defaultdict
from model.state_manager import StateManager
from test.interface.RL_coverage import RLCoverage
import numpy as np
import matplotlib

matplotlib.use('agg')
from matplotlib import pyplot as plt
import itertools

from saxpy.sax import ts_to_string
from saxpy.alphabet import cuts_for_asize
from saxpy.paa import paa


class TemporalCoverage(RLCoverage):
    def __init__(self, layer, model_name, state_manager: StateManager, symbols, seq, mean, std):
        self.name = "TemporalCoverage"

        # Thresholds
        self.mean = mean
        self.std = std

        self.plt_x = []
        self.plt_y = []
        self.fr_plt_x = []
        self.fr_plt_y = []

        self.layer = layer
        self.model_name = model_name
        self.state_manager = state_manager
        self.symbols = symbols
        self.seq = seq
        self.seq_len = int(self.seq[1]) + 1 - int(self.seq[0])

        self.hidden = None

        self.feature_length = 0
        self.__init_feature_length()

        self.covered_dict = defaultdict(bool)
        self.__init_covered_dict()
        self.activates = []

    def __init_feature_length(self):
        t1 = int(self.seq[0])
        t2 = int(self.seq[1])
        self.indices = slice(t1, t2 + 1)
        alpha_list = [chr(i) for i in range(97, 97 + int(self.symbols))]
        symbol = ''.join(alpha_list)
        self.feature = list(itertools.product(symbol, repeat=t2 - t1 + 1))
        self.feature_length = len(self.feature)

    def __init_covered_dict(self):
        for index in range(self.feature_length):
            self.covered_dict[index] = False

    def get_activation(self):
        alpha1 = np.sum(np.where(self.hidden > 0, self.hidden, 0), axis=2)
        alpha2 = np.sum(np.where(self.hidden < 0, self.hidden, 0), axis=2)
        alpha = np.abs(alpha1 + alpha2)
        return alpha

    def calculate_coverage(self):
        covered_number = 0

        for index in range(self.feature_length):
            if self.covered_dict[index] is True:
                covered_number += 1

        return covered_number, covered_number / float(self.feature_length)

    def update_features(self, data):
        self.hidden = self.state_manager.get_hidden_state([data])
        activation = self.get_activation()
        dat_znorm = (activation[:, self.indices] - self.mean) / self.std
        dat_znorm = [paa(item, self.seq_len) for item in dat_znorm]

        features = [tuple(ts_to_string(item, cuts_for_asize(self.symbols))) for item in dat_znorm]

        for feature in features:
            if feature in self.feature:
                index = self.feature.index(feature)
                self.covered_dict[index] = True

    def update_graph(self, num_samples):
        _, coverage = self.calculate_coverage()
        self.plt_x.append(num_samples)
        self.plt_y.append(coverage)

    @staticmethod
    def calculate_variation(data):
        pass

    def update_frequency_graph(self):
        pass

    def display_graph(self):
        plt.plot(self.plt_x, self.plt_y)
        plt.xlabel('# of generated samples')
        plt.ylabel('coverage')
        plt.title('Temporal Coverage of ' + self.layer.name)
        plt.savefig('output/' + self.model_name + '/' + self.layer.name + '_tc.png')
        plt.clf()

    def display_frequency_graph(self):
        pass

    def display_stat(self):
        _, coverage = self.calculate_coverage()

        f = open('output/%s_%s_spc.txt' % (self.model_name, self.layer.name), 'w')
        f.write('coverage: %f\n' % coverage)
        f.close()

    def get_name(self):
        return self.name

    def save_feature(self):
        pass