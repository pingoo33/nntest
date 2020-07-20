import numpy as np

from data.interface.oracle import Oracle


class OracleEinsum(Oracle):
    def __init__(self, radius):
        self.measurement = "ij,ij->i"
        self.radius = radius

    def pass_oracle(self, src, dest):
        distance = src - dest
        n = np.count_nonzero(distance)
        return np.sqrt(np.einsum(self.measurement, distance, distance)) / float(n) <= self.radius

    def measure(self, src, dest):
        distance = src - dest
        return np.sqrt(np.einsum(self.measurement, distance, distance))
