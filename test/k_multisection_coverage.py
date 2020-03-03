from collections import defaultdict
from test.interface.coverage import Coverage


class KMultisectionCoverage(Coverage):
    def __init__(self, size):
        self.neuron_frequency = defaultdict(int)
        self.plt_x = [[]] * size
        self.plt_y = [[]] * size
        pass

    def calculate_coverage(self):
        pass

    def update_features(self):
        pass

    def update_graph(self):
        pass

    def update_frequency_graph(self):
        pass

    def display_graph(self):
        pass

    def display_frequency_graph(self):
        pass