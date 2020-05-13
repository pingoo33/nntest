import numpy as np
from keras.datasets import mnist
from data.interface.data_manager import DataManager
from data.interface.mutant_callback import MutantCallback
from keras.utils import to_categorical


class MnistData(DataManager):
    def __init__(self, mutant_callback: MutantCallback):
        self.mutant_callback = mutant_callback
        self.x_train = None
        self.y_train = None
        self.x_test = None
        self.y_test = None
        self.raw = []
        self.num_adv = 0
        self.num_samples = 0
        self.perturbations = []
        self.scaler = None
        self.load_data()

    def load_data(self):
        (self.x_train, self.y_train), (self.x_test, self.y_test) = mnist.load_data()
        self.x_train = self.x_train.astype('float32') / 255
        self.x_test = self.x_test.astype('float32') / 255
        self.y_train = to_categorical(self.y_train)
        self.y_test = to_categorical(self.y_test)

    def mutant_data(self, data):
        new_data = self.mutant_callback.mutant_data(data)
        return new_data, None

    def get_train_data(self):
        return self.x_train, self.y_train

    def get_test_data(self):
        return self.x_test, self.y_test

    def get_num_samples(self):
        return self.num_samples

    # def update_sample(self, output2, output1, m, o):
    def update_sample(self):
        # error = abs(output2 - output1)

        # if error >= 0.0001 and o == True:
        #     self.num_adv += 1s
        #     self.perturbations.append(m)
        self.num_samples += 1
        # self.display_success_rate()
