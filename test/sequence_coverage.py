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


class SequenceCoverage(RLCoverage):
    def __init__(self, layer, model_name, state_manager: StateManager, symbols, seq):
        self.plt_x = []
        self.plt_y_p = []
        self.plt_y_n = []
        self.fr_plt_x = []
        self.fr_plt_y_p = []
        self.fr_plt_y_n = []

        self.layer = layer
        self.model_name = model_name
        self.state_manager = state_manager
        self.symbols = symbols
        self.seq = seq
        self.hidden = None
        self.__init_feature()

        self.covered_dict_p = defaultdict(bool)
        self.covered_dict_n = defaultdict(bool)
        self.frequency_dict_p = defaultdict(int)
        self.frequency_dict_n = defaultdict(int)
        self.__init_covered_dict()
        self.__init_frequency_dict()

    def __init_feature(self):
        t1 = int(self.seq[0])
        t2 = int(self.seq[1])
        self.indices = slice(t1, t2 + 1)
        alpha_list = [chr(i) for i in range(97, 97 + int(self.symbols))]
        symbol = ''.join(alpha_list)
        self.feature_p = list(itertools.product(symbol, repeat=t2 - t1 + 1))
        self.feature_n = list(itertools.product(symbol, repeat=t2 - t1 + 1))
        self.total_feature = len(self.feature_p)

    def __init_covered_dict(self):
        for index in range(self.total_feature):
            self.covered_dict_p[index] = False
            self.covered_dict_n[index] = False

    def __init_frequency_dict(self):
        for index in range(self.total_feature):
            self.frequency_dict_p[index] = 0
            self.frequency_dict_n[index] = 0

    def get_activation(self):
        hidden = self.hidden
        alpha1 = np.sum(np.where(hidden > 0, hidden, 0), axis=1)
        alpha2 = np.sum(np.where(hidden < 0, hidden, 0), axis=1)
        return alpha1, alpha2

    def calculate_coverage(self):
        covered_number_p = 0
        covered_number_n = 0

        for index in range(self.total_feature):
            if self.covered_dict_p[index] is True:
                covered_number_p += 1
            if self.covered_dict_n[index] is True:
                covered_number_n += 1

        return covered_number_p, covered_number_p / float(self.total_feature)\
            , covered_number_n, covered_number_n / float(self.total_feature)

    def update_features(self, data):
        self.hidden = self.state_manager.get_hidden_state(data)
        activation_p, activation_n = self.get_activation()
        dat_znorm_p = znorm(activation_p[self.indices])
        dat_znorm_n = znorm(activation_n[self.indices])
        sym_rep_p = ts_to_string(dat_znorm_p, cuts_for_asize(self.symbols))
        sym_rep_n = ts_to_string(dat_znorm_n, cuts_for_asize(self.symbols))
        feature_p = tuple(sym_rep_p)
        feature_n = tuple(sym_rep_n)

        if feature_p in self.feature_p:
            index = self.feature_p.index(feature_p)
            self.covered_dict_p[index] = True
            self.frequency_dict_p[index] += 1

        if feature_n in self.feature_n:
            index = self.feature_n.index(feature_n)
            self.covered_dict_n[index] = True
            self.frequency_dict_n[index] += 1

    def update_graph(self, num_samples):
        _, coverage_p, _, coverage_n = self.calculate_coverage()
        self.plt_x.append(num_samples)
        self.plt_y_p.append(coverage_p)
        self.plt_y_n.append(coverage_n)
        print("%s layer positive sequence coverage : %.8f" % (self.layer.name, coverage_p))
        print("%s layer negative sequence coverage : %.8f" % (self.layer.name, coverage_n))

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
            self.fr_plt_y_p.append(self.frequency_dict_p[index])
            self.fr_plt_y_n.append(self.frequency_dict_n[index])

    def display_graph(self):
        plt.plot(self.plt_x, self.plt_y_p)
        plt.xlabel('# of generated samples')
        plt.ylabel('coverage')
        plt.title('Positive Sequence Coverage of ' + self.layer.name)
        plt.savefig('output/' + self.model_name + '/' + self.layer.name + '_spc.png')
        plt.clf()

        plt.plot(self.plt_x, self.plt_y_n)
        plt.xlabel('# of generated samples')
        plt.ylabel('coverage')
        plt.title('Negative Sequence Coverage of ' + self.layer.name)
        plt.savefig('output/' + self.model_name + '/' + self.layer.name + '_snc.png')
        plt.clf()

    def display_frequency_graph(self):
        n_groups = len(self.fr_plt_x)
        index = np.arange(n_groups)

        plt.bar(index, self.fr_plt_y_p, align='center')

        plt.xlabel('features')
        plt.ylabel('activation counts')
        plt.title(self.layer.name + ' Frequency')
        plt.xlim(-1, n_groups)
        plt.savefig('output/' + self.model_name + '/' + self.layer.name + '_spc_Frequency.png')
        plt.clf()

        plt.bar(index, self.fr_plt_y_n, align='center')

        plt.xlabel('features')
        plt.ylabel('number of activation')
        plt.title(self.layer.name + ' Frequency')
        plt.xlim(-1, n_groups)
        plt.savefig('output/' + self.model_name + '/' + self.layer.name + '_snc_Frequency.png')
        plt.clf()
