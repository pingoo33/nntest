from data.interface.data_distribution_manager import DataDistribution
from data.interface.mutant_callback import MutantCallback
import numpy as np


class NormalMutantCallback(MutantCallback):
    def __init__(self, distribution: DataDistribution):
        self.distribution = distribution

    def mutant_data(self, data):
        mean, std = self.distribution.get_distribution(data)

        new_data = []
        for i in range(12):
            new_data.append(np.random.normal(loc=mean[i], scale=std[i]))

        return np.array(new_data)
