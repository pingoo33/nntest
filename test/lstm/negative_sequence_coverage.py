from collections import defaultdict
from model.state_manager import StateManager
from test.interface.RL_coverage import RLCoverage
import numpy as np
import matplotlib

matplotlib.use('agg')
from matplotlib import pyplot as plt
from saxpy.alphabet import cuts_for_asize
from saxpy.sax import ts_to_string
from saxpy.znorm import znorm
import itertools


class NegativeSequenceCoverage(RLCoverage):
    def save_feature(self):
        pass

    def __init__(self, layer, model_name, state_manager: StateManager, symbols, seq):
        self.name = "NegativeSequenceCoverage"
        self.plt_x = []
        self.plt_y = []

        self.layer = layer
        self.model_name = model_name
        self.state_manager = state_manager
        self.symbols = symbols
        self.seq = seq
        self.hidden = None
        self.__init_feature()

        self.covered_dict = defaultdict(bool)
        self.__init_covered_dict()

    def __init_feature(self):
        t1 = int(self.seq[0])
        t2 = int(self.seq[1])
        self.indices = slice(t1, t2 + 1)
        alpha_list = [chr(i) for i in range(97, 97 + int(self.symbols))]
        symbol = ''.join(alpha_list)
        self.feature = list(itertools.product(symbol, repeat=t2 - t1 + 1))
        self.total_feature = len(self.feature)

    def __init_covered_dict(self):
        for index in range(self.total_feature):
            self.covered_dict[index] = False

    def get_activation(self):
        hidden = self.hidden
        alpha = np.sum(np.where(hidden < 0, hidden, 0), axis=1)
        return alpha

    def calculate_coverage(self):
        covered_number = 0

        for index in range(self.total_feature):
            if self.covered_dict[index] is True:
                covered_number += 1

        return covered_number, covered_number / float(self.total_feature)

    def update_features(self, data):
        self.hidden = self.state_manager.get_hidden_state(data)
        activation = self.get_activation()
        dat_znorm = znorm(activation[self.indices])
        sym_rep = ts_to_string(dat_znorm, cuts_for_asize(self.symbols))
        feature = tuple(sym_rep)

        if feature in self.feature:
            index = self.feature.index(feature)
            self.covered_dict[index] = True

    def update_graph(self, num_samples):
        _, coverage = self.calculate_coverage()
        self.plt_x.append(num_samples)
        self.plt_y.append(coverage)
        # print("%s layer negative sequence coverage : %.8f" % (self.layer.name, coverage_n))

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
        pass

    def display_graph(self):
        plt.plot(self.plt_x, self.plt_y)
        plt.xlabel('# of generated samples')
        plt.ylabel('coverage')
        plt.title('Negative Sequence Coverage of ' + self.layer.name)
        plt.savefig('output/' + self.model_name + '/' + self.layer.name + '_snc.png')
        plt.clf()

    def display_frequency_graph(self):
        pass

    def display_stat(self):
        _, coverage = self.calculate_coverage()

        f = open('output/%s_%s_snc.txt' % (self.model_name, self.layer.name), 'w')
        f.write('coverage: %f\n' % coverage)
        f.close()

    def get_name(self):
        return self.name
