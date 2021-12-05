#! /bin/bash

cd ../pytracking/tracking_results/fcot
rm */*.pkl

cd ../../util_script

python pack_got10k_results.py treg treg_got
