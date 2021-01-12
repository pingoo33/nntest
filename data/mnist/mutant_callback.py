from data.interface.mutant_callback import MutantCallback
import random
import numpy as np
from model.interface.model_manager import ModelManager
import tensorflow.keras.backend as K


def root_mean_squared_error(y_true, y_pred):
    return K.sqrt(K.mean(K.square(y_pred - y_true), axis=-1))


class MnistMutantCallback(MutantCallback):
    def __init__(self, model_manager: ModelManager):
        self.model_manager = model_manager

    def mutant_data(self, data):
        epsilon = random.uniform(0.05, 1)
        step = 2
        new_data = data

        for s in range(step):
            last_activation = self.model_manager.get_prob(new_data)
            gradients = self.model_manager.get_gradients(np.array([new_data]), np.array([last_activation]))

            if np.shape(new_data) != np.shape(gradients):
                continue

            temp = new_data + epsilon * np.sign(gradients)
            temp = np.clip(temp, 0, 1)

            if np.array_equal(temp, new_data):
                print('Gradients are too small')
                return None

            new_data = temp

        return new_data
