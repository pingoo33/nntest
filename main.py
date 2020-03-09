import time
import argparse
import re

from temperature import *


def main():
    parser = argparse.ArgumentParser(description='testing for neural network')
    parser.add_argument('--model', dest='model_name', default='temperature', help='')
    parser.add_argument('--seed', dest='seed_num', default='2000', help='')
    parser.add_argument('--threshold_nc', dest='threshold_tc', default='0', help='')
    parser.add_argument('--sec_kmnc', dest='sec_kmnc', default='1', help='')
    parser.add_argument('--threshold_cc', dest='threshold_cc', default='5', help='')
    parser.add_argument('--threshold_gc', dest='threshold_gc', default='0.705', help='')
    parser.add_argument('--symbols_sq', dest='symbols_sq', default='2', help='')
    parser.add_argument('--seq', dest='seq', default='[7,11]', help='')
    parser.add_argument('--mode', dest='mode', default='test', help='')

    args = parser.parse_args()

    model_name = args.model_name
    seed = int(args.seed_num)
    threshold_tc = int(args.threshold_tc)
    sec_kmnc = int(args.sec_kmnc)
    threshold_cc = int(args.threshold_cc)
    threshold_gc = float(args.threshold_gc)
    symbols_sq = int(args.symbols_sq)
    seq = args.seq
    seq = re.findall(r"\d+\.?\d*", seq)
    mode = args.mode

    if model_name == 'temperature':
        if mode == 'train':
            train(model_name)
        else:
            test(model_name, seed, threshold_tc, sec_kmnc, threshold_cc, threshold_gc, symbols_sq, seq)


if __name__ == "__main__":
    start_time = time.time()
    main()
    print("--- %s seconds ---" % (time.time() - start_time))