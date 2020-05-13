from data.interface.mutant_callback import MutantCallback
import random
import numpy as np
from model.interface.model_manager import ModelManager


class MnistMutantCallback(MutantCallback):
    def __init__(self, model_manager: ModelManager):
        self.model_manager = model_manager

    def mutant_data(self, data):
        f, nodes_names = self.model_manager.get_gradients_function()

        epsilon = random.uniform(0.05, 1)
        step = random.randint(1, 5)
        last_activation = self.model_manager.get_prob(data)

        return self.get_next_input_by_gradient(f, nodes_names, epsilon, data, last_activation, step)

    def get_next_input_by_gradient(self, f, nodes_names, epsilon, data, last_activation, step):
        gd = self.model_manager.cal_gradient(f, np.array([data]), np.array([last_activation]), nodes_names)
        gd = np.squeeze(list(gd.values())[0])
        if np.shape(data) != np.shape(gd):
            step = step - 1
            if step <= 0:
                print("found a test case of shape %s!" % (str(data.shape)))
                return data
            else:
                return self.get_next_input_by_gradient(f, nodes_names, epsilon, data, last_activation, step)

        new_test = data + epsilon * np.sign(gd)
        new_test = np.clip(new_test, 0, 1)
        last_activation = self.model_manager.get_prob(data)
        step = step - 1

        if np.array_equal(data, new_test):
            print("Gradients are too small")
            print("-----------------------------------------------------")
            return None
        elif step <= 0:
            print("found a test case of shape %s!" % (str(new_test.shape)))
            return new_test
        else:
            return self.get_next_input_by_gradient(f, nodes_names, epsilon, new_test, last_activation, step)
