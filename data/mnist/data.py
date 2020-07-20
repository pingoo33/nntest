from keras.datasets import mnist
from keras.utils import to_categorical

from data.interface.data_manager import DataManager
from data.interface.mutant_callback import MutantCallback
from data.interface.oracle import Oracle


class MnistData(DataManager):
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

    def update_sample(self, src_label, dest_label, src, dest):
        if src_label != dest_label and self.oracle.pass_oracle(src, dest):
            self.num_adv += 1
            self.advs.append(dest)

        self.num_samples += 1
        self.display_success_rate()

    def display_samples(self):
        print("%s samples are considered" % self.num_samples)

    def display_success_rate(self):
        print("%s samples, within which there are %s adversarial examples" % (self.num_samples, self.num_adv))
        print("the rate of adversarial examples is %.2f\n" % (self.num_adv / self.num_samples))

