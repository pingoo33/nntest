import random
from tqdm import tqdm
import numpy as np

from data.interface.data_manager import DataManager
from model.interface.model_manager import ModelManager
from model.state_manager import StateManager
from model.threshold_manager import ThresholdManager
from test.fc.boundary_coverage import NeuronBoundaryCoverage
from test.fc.sign_sign_coverage import SignSignCoverage
from test.lstm.boundary_coverage import BoundaryCoverage
from test.lstm.cell_coverage import CellCoverage
from test.lstm.gate_coverage import GateCoverage
from test.fc.k_multisection_coverage import KMultisectionCoverage
from test.fc.threshold_coverage import ThresholdCoverage
from test.fc.top_k_coverage import TopKCoverage
from test.fc_pattern.top_k_pattern_coverage import TopKPatternCoverage
from test.lstm.negative_sequence_coverage import NegativeSequenceCoverage
from test.lstm.positive_sequence_coverage import PositiveSequenceCoverage
from test.lstm.stepwise_coverage import StepWiseCoverage
from test.lstm.temporal_coverage import TemporalCoverage


class TestNN:
    def __init__(self, data_manager: DataManager, model_manager: ModelManager, seed):
        self.data_manager = data_manager
        self.model_manager = model_manager

        self.seed = seed

    def kfold_train(self, fold_size):
        (x_train, y_train) = self.data_manager.get_train_data()
        (x_test, y_test) = self.data_manager.get_test_data()
        self.model_manager.kfold_train_model(fold_size, x_train, y_train, x_test, y_test)

    def train(self):
        (x_train, y_train) = self.data_manager.get_train_data()
        (x_test, y_test) = self.data_manager.get_test_data()
        self.model_manager.train_model(x_train, y_train, x_test, y_test)

    def __mutant_data_process(self, coverage_set, target_data):
        pbar = tqdm(range(len(target_data)), total=len(target_data))
        for i in pbar:
            # before_output = self.model_manager.get_prob(target_data[i])
            # generated_data, _ = self.data_manager.mutant_data(target_data[i])
            #
            # after_output = self.model_manager.get_prob(generated_data)
            # self.data_manager.update_sample(before_output, after_output, target_data[i], generated_data)

            for coverage in coverage_set:
                if not (target_data[i] is None):
                    coverage.update_features(target_data[i])
                    coverage.update_graph(self.data_manager.get_num_samples())

            pbar.set_description("samples: %d, advs: %d"
                                 % (self.data_manager.get_num_samples(), self.data_manager.get_num_advs()))
        for coverage in coverage_set:
            coverage.save_feature()
            coverage.update_frequency_graph()
            coverage.display_graph()
            coverage.display_frequency_graph()
            coverage.display_stat()

        for coverage in coverage_set:
            _, result = coverage.calculate_coverage()
            print("%s : %.5f" % (coverage.get_name(), result))

    def lstm_test(self, threshold_bc, threshold_sc, symbols_sq, seq):
        target_data = self.data_manager.x_train[
            random.sample(range(np.shape(self.data_manager.x_train)[0]), self.seed)]

        self.model_manager.load_model()
        model_name = self.model_manager.model_name

        indices, lstm_layers = self.model_manager.get_lstm_layer()

        init_data = target_data[15]
        layer = lstm_layers[-1]
        state_manager = StateManager(self.model_manager, indices[-1])

        mean_TC, std_TC, max_SC, min_SC, max_BC, min_BC = state_manager.aggregate_inf(target_data, seq)

        coverage_set = [BoundaryCoverage(layer, model_name, state_manager, threshold_bc, min_BC, max_BC, init_data),
                        StepWiseCoverage(layer, model_name, state_manager, threshold_sc, min_SC, max_SC, init_data),
                        TemporalCoverage(layer, model_name, state_manager, symbols_sq, seq, mean_TC, std_TC)]
        # coverage_set = [CellCoverage(layer, model_name, state_manager, threshold_cc, init_data),
        #                 GateCoverage(layer, model_name, state_manager, threshold_gc, init_data),
        #                 PositiveSequenceCoverage(layer, model_name, state_manager, symbols_sq, seq),
        #                 NegativeSequenceCoverage(layer, model_name, state_manager, symbols_sq, seq)]
        # coverage_set = [CellCoverage(layer, model_name, state_manager, threshold_cc, init_data),
        #                 GateCoverage(layer, model_name, state_manager, threshold_gc, init_data)]

        self.__mutant_data_process(coverage_set, target_data)

    def fc_test(self, threshold_tc, sec_kmnc, size_tkc):
        target_data = self.data_manager.x_train[
            random.sample(range(np.shape(self.data_manager.x_train)[0]), self.seed)]

        self.model_manager.load_model()
        other_layers = self.model_manager.get_fc_layer()

        for layer in other_layers:
            # threshold_manager = ThresholdManager(self.model_manager, layer, self.data_manager.x_train)
            # coverage_set = [ThresholdCoverage(layer, self.model_manager, threshold_tc),
            #                 KMultisectionCoverage(layer, self.model_manager, threshold_manager, sec_kmnc),
            #                 NeuronBoundaryCoverage(layer, self.model_manager, threshold_manager),
            #                 TopKCoverage(layer, self.model_manager, size_tkc)]
            coverage_set = [ThresholdCoverage(layer, self.model_manager, threshold_tc)]

            self.__mutant_data_process(coverage_set, target_data)

    def link_test(self):
        target_indices = random.sample(range(np.shape(self.data_manager.x_train)[0]), self.seed)
        target_before_data = self.data_manager.x_train[target_indices]
        target_after_data = self.data_manager.x_train[target_indices]

        target_data = []

        for b_index, b_data in enumerate(target_before_data):
            for a_index, a_data in enumerate(target_after_data):
                if b_index == a_index:
                    continue
                target_data.append([b_data, a_data])

        self.model_manager.load_model()
        other_layers = self.model_manager.get_fc_layer()

        for layers in other_layers:
            coverage_set = [SignSignCoverage(layers[0], layers[1], self.model_manager)]

            self.__mutant_data_process(coverage_set, target_data)

    def pattern_test(self, size_tkpc):
        target_data = self.data_manager.x_train[
            random.sample(range(np.shape(self.data_manager.x_train)[0]), self.seed)]

        self.model_manager.load_model()
        coverage_set = [TopKPatternCoverage(self.model_manager, size_tkpc)]

        self.__mutant_data_process(coverage_set, target_data)
