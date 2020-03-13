from data.interface.data_distribution_manager import DataDistribution
import os
import numpy as np
import pandas as pd


class TemperatureDistribution(DataDistribution):
    def __init__(self):
        self.means = []
        self.stds = []
        self.load_distribution()

    def load_distribution(self):
        train_data = pd.read_csv("./dataset/train_data.csv", sep=',')
        train_data = np.array(train_data)
        raw = train_data[:, 3:]

        sort_data = [[]] * 4

        for data in raw:
            sort_data[self.get_index(data)].append(data)

        model_data_by_column = []
        for data_list in sort_data:
            temp = []
            for column in range(12):
                temp_column = []
                for data in data_list:
                    temp_column.append(data[column])
                temp.append(temp_column)
            model_data_by_column.append(temp)

        model_data_by_column = np.array(model_data_by_column)

        means_by_file = []
        stds_by_file = []
        for file in range(4):
            means = []
            stds = []
            for mut_column in range(12):
                means.append(np.mean(model_data_by_column[file][mut_column]))
                stds.append(np.std(model_data_by_column[file][mut_column]))
            means_by_file.append(means)
            stds_by_file.append(stds)

        self.means = means_by_file
        self.stds = stds_by_file

    @staticmethod
    def get_index(data):
        temperature = data[0]

        if 15 < temperature < 25:
            target_f_idx = 0
        elif temperature >= 25:
            target_f_idx = 1
        elif -5 < temperature < 15:
            target_f_idx = 2
        else:
            target_f_idx = 3

        return target_f_idx

    def get_distribution(self, data):
        f_idx = self.get_index(data)

        return self.means[f_idx], self.stds[f_idx]
