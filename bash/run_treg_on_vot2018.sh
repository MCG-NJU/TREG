#! /bin/bash

cd ../pytracking

python run_tracker.py treg treg_vot2018 --dataset vot \
      --threads 0  --params__lamda_18 1.0 \
      --cuda_id 0 --params__merge_rate_18 0.8 --params__search_area_scale 4.1 --params__lamda_72 0.4
