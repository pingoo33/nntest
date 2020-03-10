import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler
import random
from data.interface.data_manager import DataManager
from data.interface.mutant_callback import MutantCallback


class TemperatureData(DataManager):
    def __init__(self, mutant_callback: MutantCallback):
        self.mutant_callback = mutant_callback
        self.num_samples = None
        self.x_train = None
        self.y_train = None
        self.x_test = None
        self.y_test = None
        self.raw_data = []
        self.num_adv = 0
        self.num_samples = 0
        self.perturbations = []
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
        self.raw_data = train_data

        self.scaler = RobustScaler()
        self.scaler.fit(self.raw_data[:, 3:])
        normalize = self.scaler.transform(self.raw_data[:, 3:])

        DF_data = normalize
        Adata_1 = np.array(DF_data)

        raw_x = Adata_1
        raw_y = Adata_1[:, -1:]

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
        new_raw_data = np.append(self.raw_data, new_axis_data, axis=0)
        scaler = RobustScaler()
        scaler.fit(new_raw_data)
        new_normalize = scaler.transform(new_raw_data)
        return new_normalize[-1]

    def mutant_data(self, data):
        random_idx = random.randrange(0, 12)
        selected_data = data[random_idx]
        origin_data = self.scaler.inverse_transform(selected_data[np.newaxis, :])[0]
        new_data = np.array(origin_data)

        for i in range(11):
            new_data[i] = self.mutant_callback.mutant_data(origin_data)

        data[random_idx] = np.array(new_data)
        return data, new_data

    def get_num_samples(self):
        return self.num_samples

    # def update_sample(self, output2, output1, m, o):
    def update_sample(self):
        # error = abs(output2 - output1)

        # if error >= 0.0001 and o == True:
        #     self.num_adv += 1s
        #     self.perturbations.append(m)
        self.num_samples += 1
        self.display_success_rate()

    def display_samples(self):
        print("%s samples are considered" % self.num_samples)

    def display_success_rate(self):
        print("%s samples, within which there are %s adversarial examples" % (self.num_samples, self.num_adv))
        print("the rate of adversarial examples is %.2f\n" % (self.num_adv / self.num_samples))

    def display_perturbations(self):
        if self.num_adv > 0:
            print(
                "the average perturbation of the adversarial examples is %s" % (sum(self.perturbations) / self.num_adv))
            print("the smallest perturbation of the adversarial examples is %s" % (min(self.perturbations)))
