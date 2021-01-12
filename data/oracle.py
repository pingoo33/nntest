import numpy as np

from data.interface.oracle import Oracle


class NormalOracle(Oracle):
    def __init__(self, radius):
        self.radius = radius

    def pass_oracle(self, src, dest):
        n = np.count_nonzero(src - dest)
        return np.linalg.norm(src - dest, ord=2) / float(n) <= self.radius

    def measure(self, src, dest):
        return np.linalg.norm(src - dest, ord=2)
