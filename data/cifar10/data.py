from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical

from data.interface.data_manager import DataManager
from data.interface.mutant_callback import MutantCallback
from data.interface.oracle import Oracle


class Cifar10Data(DataManager):
    def __init__(self, mutant_callback: MutantCallback, oracle: Oracle):
        super().__init__()
        self.mutant_callback = mutant_callback
        self.oracle = oracle
        self.raw = []
        self.num_adv = 0
        self.advs = []
        self.num_samples = 0
        self.scaler = None
        self.load_data()

    def load_data(self):
        (self.x_train, self.y_train), (self.x_test, self.y_test) = cifar10.load_data()
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

    def update_sample(self, src_label, dest_label, src, dest):
        if src_label != dest_label and self.oracle.pass_oracle(src, dest):
            self.num_adv += 1
            self.advs.append(dest)

        self.num_samples += 1

    def get_num_advs(self):
        return self.num_adv

    def save_advs(self):
        pass
