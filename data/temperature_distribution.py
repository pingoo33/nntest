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
        csv_file_list = []

        for file in os.listdir("./dataset/temperature"):
            if file.endswith(".csv"):
                data = pd.read_csv("./dataset/temperature/" + str(file), sep=',')
                csv_file_list.append(np.array(data, dtype=float))

        file_list = []
        for i in range(len(csv_file_list)):
            DF_data = csv_file_list[i]
            Adata_1 = np.array(DF_data)
            file_list.append(Adata_1[:, 3:-1])

        model_data_by_column = []
        for data_list in file_list:
            file_by_column = []
            for mut_column in range(11):
                data_by_column = []
                for d in data_list:
                    data_by_column.append(d[mut_column])
                file_by_column.append(data_by_column)
            model_data_by_column.append(file_by_column)

        model_data_by_column = np.array(model_data_by_column)

        means_by_file = []
        stds_by_file = []
        for file in range(4):
            means = []
            stds = []
            for mut_column in range(11):
                means.append(np.mean(model_data_by_column[file][mut_column]))
                stds.append(np.std(model_data_by_column[file][mut_column]))
            means_by_file.append(means)
            stds_by_file.append(stds)

        self.means = means_by_file
        self.stds = stds_by_file

    @staticmethod
    def get_index(data):
        month = data[0]

        if month == 12 or 1 <= month <= 2:
            target_f_idx = 0
        elif 3 <= month <= 5:
            target_f_idx = 1
        elif 6 <= month <= 8:
            target_f_idx = 2
        else:
            target_f_idx = 3

        return target_f_idx

    def get_distribution(self, data):
        f_idx = self.get_index(data)

        return self.means[f_idx], self.stds[f_idx]
