from collections import defaultdict
import numpy as np

from model.interface.model_manager import ModelManager


class ThresholdManager:
    def __init__(self, model_manager: ModelManager, data_set):
        self.model_manager = model_manager

        self.max_threshold_dict = defaultdict(float)
        self.min_threshold_dict = defaultdict(float)
        self.__init_threshold_dict(data_set)

    def __init_threshold_dict(self, data_set):
        for layer_index, layer in enumerate(self.model_manager.model.layers):
            if 'input' in layer.name or 'concatenate' in layer.name:
                continue
            for index in range(layer.output_shape[-1]):
                self.max_threshold_dict[(layer.name, index)] = 0
                self.min_threshold_dict[(layer.name, index)] = 0

        layers = [layer for layer_index, layer in enumerate(self.model_manager.model.layers)
                  if 'concatenate' not in layer.name and 'input' not in layer.name]

        for data in data_set:
            for layer in layers:
                output = self.model_manager.get_intermediate_output(layer, data)
                for num_neuron in range(output.shape[-1]):
                    if np.mean(output[..., num_neuron]) > self.max_threshold_dict[(layer.name, num_neuron)]:
                        self.max_threshold_dict = np.mean(output[..., num_neuron])
                    elif np.mean(output[..., num_neuron]) < self.min_threshold_dict[(layer.name, num_neuron)]:
                        self.min_threshold_dict = np.mean(output[..., num_neuron])

    def get_max_threshold(self, layer_name, index):
        return self.max_threshold_dict[(layer_name, index)]

    def get_min_threshold(self, layer_name, index):
        return self.min_threshold_dict[(layer_name, index)]
