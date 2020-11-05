import random
import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler

from data.interface.data_manager import DataManager
from data.interface.mutant_callback import MutantCallback
from data.interface.oracle import Oracle


class TemperatureData(DataManager):
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

    def get_train_data(self):
        return self.x_train, self.y_train

    def get_test_data(self):
        return self.x_test, self.y_test

    def load_data(self):
        n_seq = 12

        train_data = pd.read_csv("./dataset/train_data.csv", sep=',')

        data_x = []
        data_y = []

        train_data = np.array(train_data)
        self.raw = train_data[:, 3:]

        # self.scaler = RobustScaler()
        # self.scaler.fit(self.raw)
        # normalize = self.scaler.transform(self.raw)
        #
        # DF_data = normalize
        # Adata_1 = np.array(DF_data)

        raw_x = self.raw
        raw_y = self.raw[:, -1:]

        for j in range((len(raw_y) - n_seq)):
            _x = raw_x[j:j + n_seq]
            _y = raw_y[j + n_seq]
            data_x.append(_x)
            data_y.append(_y)

        train_size = int(len(data_y) * 0.6)
        self.x_train, self.x_test = np.array(data_x[0:train_size]), np.array(data_x[train_size: len(data_x)])
        self.y_train, self.y_test = np.array(data_y[0:train_size]), np.array(data_y[train_size: len(data_x)])

    def normalize(self, data):
        new_axis_data = data[np.newaxis, :]
        new_raw_data = np.append(self.raw, new_axis_data, axis=0)
        scaler = RobustScaler()
        scaler.fit(new_raw_data)
        new_normalize = scaler.transform(new_raw_data)
        return new_normalize[-1]

    def mutant_data(self, data):
        random_idx = random.randrange(0, 12)
        selected_data = data[random_idx]
        # origin_data = self.scaler.inverse_transform(selected_data[np.newaxis, :])[0]
        #
        # new_data = self.mutant_callback.mutant_data(origin_data)
        new_data = self.mutant_callback.mutant_data(selected_data)

        # normalized_new_data = self.normalize(new_data)

        new = np.array(data)
        # new[random_idx] = np.array(normalized_new_data)
        new[random_idx] = np.array(new_data)

        return new, new_data

    def get_num_samples(self):
        return self.num_samples

    def update_sample(self, src_label, dest_label, src=None, dest=None):
        if abs(src_label - dest_label) > 0.09 and self.oracle.pass_oracle(src, dest):
            self.num_adv += 1
            self.advs.append(dest)

        self.num_samples += 1

    def get_num_advs(self):
        return self.num_adv

    def save_advs(self):
        activations = pd.DataFrame(self.advs)
        activations.to_csv('temperature_advs.csv', mode='w')
