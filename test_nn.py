import random

from data.interface.data_manager import DataManager
from model.interface.model_manager import ModelManager
from model.state_manager import StateManager
from model.threshold_manager import ThresholdManager
from test.fc.boundary_coverage import BoundaryCoverage
from test.lstm.cell_coverage import CellCoverage
from test.lstm.gate_coverage import GateCoverage
from test.fc.k_multisection_coverage import KMultisectionCoverage
from test.lstm.sequence_coverage import SequenceCoverage
from test.fc.threshold_coverage import ThresholdCoverage
from test.fc.top_k_coverage import TopKCoverage
from test.fc_pattern.top_k_pattern_coverage import TopKPatternCoverage


class TestNN:
    def __init__(self, data_manager: DataManager, model_manager: ModelManager):
        self.data_manager = data_manager
        self.model_manager = model_manager

    def kfold_train(self, fold_size):
        (x_train, y_train) = self.data_manager.get_train_data()
        (x_test, y_test) = self.data_manager.get_test_data()
        self.model_manager.kfold_train_model(fold_size, x_train, y_train, x_test, y_test)

    def train(self):
        (x_train, y_train) = self.data_manager.get_train_data()
        (x_test, y_test) = self.data_manager.get_test_data()
        self.model_manager.train_model(x_train, y_train, x_test, y_test)

    def __mutant_data_process(self, coverage_set, target_data):
        for data in target_data:
            # TODO: implement oracle
            # before_output = model_manager.get_prob(data)
            generated_data, _ = self.data_manager.mutant_data(data)

            for coverage in coverage_set:
                if not (data is None):
                    # after_output = model_manager.get_prob(data)
                    self.data_manager.update_sample()

                    coverage.update_features(data)
                    coverage.update_graph(self.data_manager.get_num_samples() / len(coverage_set))

        for coverage in coverage_set:
            coverage.update_frequency_graph()
            coverage.display_graph()
            coverage.display_frequency_graph()
            coverage.display_stat()

    def __lstm_test(self, target_data, threshold_cc, threshold_gc, symbols_sq, seq):
        model = self.model_manager.model
        model_name = self.model_manager.model_name

        indices, lstm_layers = self.model_manager.get_lstm_layer()

        init_data = target_data[15]
        layer = lstm_layers[0]
        state_manager = StateManager(model, indices[0])
        coverage_set = [CellCoverage(layer, model_name, state_manager, threshold_cc, init_data),
                        GateCoverage(layer, model_name, state_manager, threshold_gc, init_data),
                        SequenceCoverage(layer, model_name, state_manager, symbols_sq, seq)]

        self.__mutant_data_process(coverage_set, target_data)

    def __fc_test(self, target_data, threshold_tc, sec_kmnc, size_tkc):
        _, other_layers = self.model_manager.get_fc_layer()

        for layer in other_layers:
            threshold_manager = ThresholdManager(self.model_manager, layer, self.data_manager.x_train)
            coverage_set = [ThresholdCoverage(layer, self.model_manager, threshold_tc),
                            KMultisectionCoverage(layer, self.model_manager, threshold_manager, sec_kmnc),
                            BoundaryCoverage(layer, self.model_manager, threshold_manager),
                            TopKCoverage(layer, self.model_manager, size_tkc)]

            self.__mutant_data_process(coverage_set, target_data)

    def __pattern_test(self, target_data, size_tkpc):
        coverage_set = [TopKPatternCoverage(self.model_manager, size_tkpc)]

        self.__mutant_data_process(coverage_set, target_data)

    def test(self, seed, threshold_tc, sec_kmnc, threshold_cc, threshold_gc, symbols_sq, seq, size_tkc, size_tkpc):
        self.model_manager.load_model()

        target_data = self.data_manager.x_train[random.sample(range(6000), seed)]

        _, other_layers = self.model_manager.get_fc_layer()

        self.__lstm_test(target_data, threshold_cc, threshold_gc, symbols_sq, seq)

        self.__fc_test(target_data, threshold_tc, sec_kmnc, size_tkc)

        self.__pattern_test(target_data, size_tkpc)
