python3 main.py --model kfold_temperature --seed 6000 --threshold_nc 0 --sec_kmnc 10 --threshold_cc 2.5 --threshold_gc 0.708 --symbols_sq 2 --seq [7,11] --size_tkc 3 --size_tkpc 2 --mode test
python3 main.py --model temperature --seed 6000 --threshold_nc 0 --sec_kmnc 10 --threshold_cc 2.5 --threshold_gc 0.708 --symbols_sq 2 --seq [7,11] --size_tkc 3 --size_tkpc 2 --mode test

python3 main.py --model kfold_mnist --seed 500 --threshold_nc 0 --sec_kmnc 10 --threshold_cc 6 --threshold_gc 0.71 --symbols_sq 2 --seq [1,5] --size_tkc 3 --size_tkpc 3 --mode test
python3 main.py --model mnist --seed 500 --threshold_nc 0 --sec_kmnc 10 --threshold_cc 6 --threshold_gc 0.71 --symbols_sq 2 --seq [1,5] --size_tkc 3 --size_tkpc 3 --mode test
