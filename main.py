import time
import argparse
import re

from data.atomic.data import AtomicData
from data.atomic.distribution import AtomicDistribution
from data.atomic.mutant_callback import AtomicMutantCallback
from data.cifar10.data import Cifar10Data
from data.cifar10.mutant_callback import Cifar10MutantCallback
from data.mnist.data import MnistData
from data.mnist.data_cnn import MnistCNNData
from data.mnist.mutant_callback import MnistMutantCallback
from data.oracle_einsum import OracleEinsum
from model.atomic import Atomic
from model.cifar10 import Cifar10
from model.mnist import Mnist
from model.mnist_cnn import MnistCNN
from model.resnet import Resnet
from test_gan import TestGAN
from test_nn import *
from model.temperature import Temperature
from data.temperature.normal_mutant_callback import NormalMutantCallback
from data.temperature.data import TemperatureData
from data.temperature.distribution import TemperatureDistribution


def main():
    parser = argparse.ArgumentParser(description='testing for neural network')
    parser.add_argument('--model', dest='model_name', default='temperature', help='')
    parser.add_argument('--seed', dest='seed_num', default='2000', help='')
    parser.add_argument('--threshold_nc', dest='threshold_tc', default='0', help='')
    parser.add_argument('--sec_kmnc', dest='sec_kmnc', default='1', help='')
    parser.add_argument('--threshold_bc', dest='threshold_bc', default='5', help='')
    parser.add_argument('--threshold_sc', dest='threshold_sc', default='0.705', help='')
    parser.add_argument('--symbols_sq', dest='symbols_sq', default='2', help='')
    parser.add_argument('--seq', dest='seq', default='[7,11]', help='')
    parser.add_argument('--size_tkc', dest='size_tkc', default='1', help='')
    parser.add_argument('--size_tkpc', dest='size_tkpc', default='1', help='')
    parser.add_argument('--fold_size', dest='fold_size', default='1', help='')
    parser.add_argument('--mode', dest='mode', default='test_lstm', help='')

    args = parser.parse_args()

    model_name = args.model_name
    seed = int(args.seed_num)
    threshold_tc = int(args.threshold_tc)
    sec_kmnc = int(args.sec_kmnc)
    threshold_bc = float(args.threshold_bc)
    threshold_sc = float(args.threshold_sc)
    symbols_sq = int(args.symbols_sq)
    seq = args.seq
    seq = re.findall(r"\d+\.?\d*", seq)
    size_tkc = int(args.size_tkc)
    size_tkpc = int(args.size_tkpc)
    fold_size = int(args.fold_size)
    mode = args.mode

    radius = 0.005
    test = None

    """
        Temperature:
        threshold_cc    = 2.5
        threshold_gc    = 0.708
        symbols_sq      = 2
        seq             = [7,11]
        
        Atomic:
        threshold_cc    = 5.11407
        threshold_gc    = 0.90093
        symbols_sq      = 2
        seq             = [5,9]
    """
    if 'temperature' in model_name:
        radius = 0.6

        data_distribution = TemperatureDistribution()
        mutant_callback = NormalMutantCallback(data_distribution)
        oracle = OracleEinsum(radius)
        data_manager = TemperatureData(mutant_callback, oracle)
        model_manager = Temperature(model_name)

        test = TestNN(data_manager, model_manager, seed)
    elif 'mnist_cnn' in model_name:
        model_manager = MnistCNN(model_name)
        mutant_callback = MnistMutantCallback(model_manager)
        oracle = OracleEinsum(radius)
        data_manager = MnistCNNData(mutant_callback, oracle)

        test = TestNN(data_manager, model_manager, seed)
    elif 'mnist' in model_name:
        model_manager = Mnist(model_name)
        mutant_callback = MnistMutantCallback(model_manager)
        oracle = OracleEinsum(radius)
        data_manager = MnistData(mutant_callback, oracle)

        test = TestNN(data_manager, model_manager, seed)
    elif 'atomic' in model_name:
        radius = 6.66128

        model_manager = Atomic(model_name)
        data_distribution = AtomicDistribution()
        mutant_callback = AtomicMutantCallback(data_distribution)
        oracle = OracleEinsum(radius)
        data_manager = AtomicData(mutant_callback, oracle)

        test = TestNN(data_manager, model_manager, seed)
    elif 'cifar10' in model_name:
        model_manager = Cifar10(model_name)
        mutant_callback = Cifar10MutantCallback(model_manager)
        oracle = OracleEinsum(radius)
        data_manager = Cifar10Data(mutant_callback, oracle)

        test = TestNN(data_manager, model_manager, seed)
    elif 'resnet' in model_name:
        model_manager = Resnet(model_name)
        mutant_callback = Cifar10MutantCallback(model_manager)
        oracle = OracleEinsum(radius)
        data_manager = Cifar10Data(mutant_callback, oracle)

        test = TestNN(data_manager, model_manager, seed)
    elif 'test_gan' in model_name:
        model_manager = MnistCNN('mnist_cnn')
        mutant_callback = MnistMutantCallback(model_manager)
        oracle = OracleEinsum(radius)
        data_manager = MnistCNNData(mutant_callback, oracle)

        test = TestGAN(data_manager, model_manager)

    if mode == 'train':
        if 'kfold' in model_name:
            test.kfold_train(fold_size)
        else:
            test.train()
    elif mode == 'test_lstm':
        test.lstm_test(threshold_bc, threshold_sc, symbols_sq, seq)
    elif mode == 'test_fc':
        test.fc_test(threshold_tc, sec_kmnc, size_tkc)
    elif mode == 'test_pattern':
        test.pattern_test(size_tkpc)


if __name__ == "__main__":
    start_time = time.time()
    main()
    print("--- %s seconds ---" % (time.time() - start_time))
