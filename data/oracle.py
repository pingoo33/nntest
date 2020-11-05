import numpy as np


class Oracle:
    def __init__(self, input, lp, radius):
        self.input = input
        self.measurement = lp
        self.radius = radius

    def pass_oracle(self, test):
        n = np.count_nonzero(self.input - test)
        return np.linalg.norm(self.input - test, ord=self.measurement) / float(n) <= self.radius

    def measure(self, test):
        return np.linalg.norm(self.input - test, ord=self.measurement)
