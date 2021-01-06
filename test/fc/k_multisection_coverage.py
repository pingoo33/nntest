from model.interface.model_manager import ModelManager
from model.threshold_manager import ThresholdManager
from test.interface.FCL_coverage import FCLCoverage
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use('agg')
from matplotlib import pyplot as plt
from matplotlib import cm


class KMultisectionCoverage(FCLCoverage):
    def save_feature(self):
        pass

    def __init__(self, layer, model_manager: ModelManager, threshold_manager: ThresholdManager, size):
        self.name = "KMultisectionCoverage"
        self.num_section = size
        self.plt_x = []
        self.plt_y = []
        self.fr_plt_x = []
        self.fr_plt_y = [[]] * size
        self.viridis = cm.get_cmap('viridis', size)

        self.layer = layer
        self.model_manager = model_manager
        self.threshold_manager = threshold_manager

        self.covered_dicts = []
        self.__init_covered_dict()
        self.frequency_dicts = []
        self.__init_frequency_dict()

    def __init_covered_dict(self):
        for section in range(self.num_section):
            temp = []
            for index in range(self.layer.output_shape[-1]):
                temp.append(False)
            self.covered_dicts.append(temp)

    def __init_frequency_dict(self):
        for section in range(self.num_section):
            temp = []
            for index in range(self.layer.output_shape[-1]):
                temp.append(0)
            self.frequency_dicts.append(temp)

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
        # print("%s layer k-multisection coverage : %.8f" % (self.layer.name, coverage))

    def update_frequency_graph(self):
        for num_neuron in range(self.layer.output_shape[-1]):
            self.fr_plt_x.append(num_neuron)

        for index in range(self.num_section):
            temp = []
            for num_neuron in range(self.layer.output_shape[-1]):
                temp.append(self.frequency_dicts[index][num_neuron])
            self.fr_plt_y[index] = temp

    def display_graph(self, name=''):
        name = name + self.name
        plt.plot(self.plt_x, self.plt_y)
        plt.xlabel('# of generated samples')
        plt.ylabel('coverage')
        plt.title('K-Multisection Coverage of ' + self.layer.name)
        plt.savefig('output/' + self.model_manager.model_name + '/' + self.layer.name + '_' + name + '.png')
        plt.clf()

    def display_frequency_graph(self, name=''):
        name = name + self.name
        self.fr_plt_y = np.array(self.fr_plt_y)
        self.fr_plt_y = np.swapaxes(self.fr_plt_y, 0, 1)
        df = pd.DataFrame(self.fr_plt_y)

        title = self.layer.name + ' Frequency of K-Multisection Coverage'
        ax = df.plot(kind='bar', stacked=True, figsize=(10, 6), legend=False, title=title,
                     xticks=([w for w in range(len(self.fr_plt_x)) if w % 10 == 0]))
        ax.set_xlabel('neuron')
        ax.set_ylabel('number of activation')
        plt.savefig('output/' + self.model_manager.model_name + '/' + self.layer.name + '_' + name + '_Frequency.png')
        plt.clf()

    def display_stat(self, name=''):
        _, coverage = self.calculate_coverage()
        mean, variation = self.calculate_variation(self.fr_plt_y)

        f = open('output/%s/%s_%s_kmnc.txt' % (self.model_manager.model_name, name, self.layer.name), 'w')
        f.write('coverage: %f\n' % coverage)
        f.write('mean: %f\n' % mean)
        f.write('variation: %f' % variation)
        f.close()

    def get_name(self):
        return self.name
