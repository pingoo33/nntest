import random

from model.state_manager import StateManager
from model.threshold_manager import ThresholdManager
from test.boundary_coverage import BoundaryCoverage
from test.cell_coverage import CellCoverage
from test.gate_coverage import GateCoverage
from test.k_multisection_coverage import KMultisectionCoverage
from test.sequence_coverage import SequenceCoverage
from test.threshold_coverage import ThresholdCoverage
from test.top_k_coverage import TopKCoverage
from test.top_k_pattern_coverage import TopKPatternCoverage


class TestNN:
    def __init__(self, data_manager, model_manager):
        self.data_manager = data_manager
        self.model_manager = model_manager

    def train(self, fold_size):
        self.model_manager.train_model(fold_size)

    def __mutant_data_process(self, coverage_set, target_data):
        for data in target_data:
            for coverage in coverage_set:
                # TODO: implement oracle
                # before_output = model_manager.get_prob(data)
                generated_data, _ = self.data_manager.mutant_data(data)

                if not (data is None):
                    # after_output = model_manager.get_prob(data)
                    self.data_manager.update_sample()

                    coverage.update_features(data)
                    coverage.update_graph(self.data_manager.get_num_samples())

        for coverage in coverage_set:
            coverage.update_frequency_graph()
            coverage.display_graph()
            coverage.display_frequency_graph()

    def __lstm_test(self, target_data, threshold_cc, threshold_gc, symbols_sq, seq):
        model = self.model_manager.model
        model_name = self.model_manager.model_name

        indices, lstm_layers = self.model_manager.get_lstm_layer()

        init_data = target_data[15]
        layer = lstm_layers[0]
        print(indices[0])
        state_manager = StateManager(model, indices[0])
        coverage_set = [CellCoverage(layer, model_name, state_manager, threshold_cc, init_data),
                        GateCoverage(layer, model_name, state_manager, threshold_gc, init_data),
                        SequenceCoverage(layer, model_name, state_manager, symbols_sq, seq)]

        self.__mutant_data_process(coverage_set, target_data)

    def __fc_test(self, target_data, threshold_tc, sec_kmnc, size_tkc, size_tkpc):
        _, other_layers = self.model_manager.get_fc_layer()

        for layer in other_layers:
            threshold_manager = ThresholdManager(self.model_manager, layer, self.data_manager.x_train)
            coverage_set = [ThresholdCoverage(layer, self.model_manager, threshold_tc),
                            KMultisectionCoverage(layer, self.model_manager, threshold_manager, sec_kmnc),
                            BoundaryCoverage(layer, self.model_manager, threshold_manager),
                            TopKCoverage(layer, self.model_manager, size_tkc)]

            self.__mutant_data_process(coverage_set, target_data)

        coverage_set = [TopKPatternCoverage(self.model_manager, size_tkpc)]

        self.__mutant_data_process(coverage_set, target_data)

    def test(self, seed, threshold_tc, sec_kmnc, threshold_cc, threshold_gc, symbols_sq, seq, size_tkc, size_tkpc):
        self.model_manager.load_model()

        target_data = self.data_manager.x_train[random.sample(range(6000), seed)]

        _, other_layers = self.model_manager.get_fc_layer()

        self.__lstm_test(target_data, threshold_cc, threshold_gc, symbols_sq, seq)

        self.__fc_test(target_data, threshold_tc, sec_kmnc, size_tkc, size_tkpc)
