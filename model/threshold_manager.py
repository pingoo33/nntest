import numpy as np
from tqdm import tqdm
from collections import defaultdict

from model.interface.model_manager import ModelManager


class ThresholdManager:
    def __init__(self, model_manager: ModelManager, layer, data_set):
        self.model_manager = model_manager
        self.layer = layer

        self.max_threshold_dict = defaultdict(float)
        self.min_threshold_dict = defaultdict(float)
        self.__init_threshold_dict(data_set)

    def __init_threshold_dict(self, data_set):
        for index in range(self.layer.output_shape[-1]):
            self.max_threshold_dict[index] = 0
            self.min_threshold_dict[index] = 0

        pbar = tqdm(range(len(data_set)), total=len(data_set))
        for i in pbar:
            output = self.model_manager.get_intermediate_output(self.layer, data_set[i])
            for num_neuron in range(output.shape[-1]):
                if np.mean(output[..., num_neuron]) > self.max_threshold_dict[num_neuron]:
                    self.max_threshold_dict[num_neuron] = float(np.mean(output[..., num_neuron]))
                elif np.mean(output[..., num_neuron]) < self.min_threshold_dict[num_neuron]:
                    self.min_threshold_dict[num_neuron] = float(np.mean(output[..., num_neuron]))
        pbar.clear()

    def get_max_threshold(self, index):
        return self.max_threshold_dict[index]

    def get_min_threshold(self, index):
        return self.min_threshold_dict[index]
