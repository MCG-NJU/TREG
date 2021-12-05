#! /bin/bash

cd ../ltr

WORKSPACE_STAGE1="/YOUR/WORKSPACE/PATH"
LOAD_MODEL_EPOCH=70

# stage1: train backbone, classifier-72, classifier-18 and regression branch (except for optimizer)
python run_training.py fcot treg_vot \
  --samples_per_epoch 30000 \
  --use_pretrained_dimp 'True' \
  --pretrained_dimp50 "../models/dimp50.pth" \
  --workspace_dir $WORKSPACE_STAGE1 \
  --lasot_rate 0.1 \
  --total_epochs 70 \
  --norm_scale_coef 2 \
  --batch_size 80 \
  --num_workers 6 \
  --devices_id 0 1 2 3 4 5 6 7 # used gpus
