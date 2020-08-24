import numpy as np

from data.interface.oracle import Oracle


class OracleEinsum(Oracle):
    def __init__(self, radius):
        self.measurement = "ij,ij->j"
        self.radius = radius

    def pass_oracle(self, src, dest):
        distance = src - dest
        return np.mean(np.sqrt(np.einsum(self.measurement, distance, distance))) <= self.radius

    def measure(self, src, dest):
        distance = src - dest
        return np.mean(np.sqrt(np.einsum(self.measurement, distance, distance)))
