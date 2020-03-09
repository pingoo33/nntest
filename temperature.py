import random

from data.normal_mutant_callback import NormalMutantCallback
from data.temerature import TemperatureData
from data.temperature_distribution import TemperatureDistribution
from model.state_manager import StateManager
from model.temperature import Temperature
from model.threshold_manager import ThresholdManager
from test.boundary_coverage import BoundaryCoverage
from test.cell_coverage import CellCoverage
from test.gate_coverage import GateCoverage
from test.k_multisection_coverage import KMultisectionCoverage
from test.sequence_coverage import SequenceCoverage
from test.threshold_coverage import ThresholdCoverage


def __get_manager(model_name):
    data_distribution = TemperatureDistribution()
    mutant_callback = NormalMutantCallback(data_distribution)
    data_manager = TemperatureData(mutant_callback)

    return data_manager, Temperature(data_manager, model_name)


def __get_layer_type(layer_name):
    return layer_name.split('_')[0]


def train(model_name):
    _, model_manager = __get_manager(model_name)
    model_manager.train_model()


def __mutant_data_process(data_manager, coverage_set, target_data):
    for data in target_data:
        for coverage in coverage_set:
            # TODO: implement oracle
            # before_output = model_manager.get_prob(data)
            generated_data, _ = data_manager.mutant_data(data)

            if not (data is None):
                # after_output = model_manager.get_prob(data)
                data_manager.update_sample()

                coverage.update_features(data)
                coverage.update_graph(data_manager.num_samples)

    for coverage in coverage_set:
        coverage.update_frequency_graph()
        coverage.display_graph()
        coverage.display_frequency_graph()


def test(model_name, seed, threshold_tc, sec_kmnc, threshold_cc, threshold_gc, symbols_sq, seq):
    data_manager, model_manager = __get_manager(model_name)
    threshold_manager = ThresholdManager(model_manager, data_manager.x_train)

    target_data = data_manager.x_train[random.sample(range(3000), seed)]

    for index, layer in enumerate(model_manager.model.layers):
        if 'input' in layer.name or 'concatenate' in layer.name:
            continue
        layer_type = __get_layer_type(layer.name)
        if layer_type == "lstm":
            init_data = target_data[15]
            state_manager = StateManager(model_manager, index)
            coverage_set = [CellCoverage(layer, model_manager, state_manager, threshold_cc, init_data),
                            GateCoverage(layer, model_manager, state_manager, threshold_gc, init_data),
                            SequenceCoverage(layer, model_manager, state_manager, symbols_sq, seq)]

            __mutant_data_process(data_manager, coverage_set, target_data)
        else:
            coverage_set = [ThresholdCoverage(layer, model_manager, threshold_tc),
                            KMultisectionCoverage(layer, model_manager, threshold_manager, sec_kmnc),
                            BoundaryCoverage(layer, model_manager, threshold_manager)]

            __mutant_data_process(data_manager, coverage_set, target_data)
