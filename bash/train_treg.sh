#! /bin/bash

cd ../ltr

WORKSPACE"/YOUR/WORKSPACE/PATH"

# train treg without DDP. our default training
#python run_training.py fcot treg \
#  --samples_per_epoch 40000 \
#  --workspace_dir $WORKSPACE \
#  --lasot_rate 1 \
#  --total_epochs 70 \
#  --norm_scale_coef 2 \
#  --batch_size 80 \
#  --num_workers 10 \
#  --devices_id 0 1 2 3 4 5 6 7  # used gpus


# train treg with DDP
# default: batch_size 10, samples_per_epoch 5000
python -m torch.distributed.launch --nproc_per_node 8  run_training.py treg treg_ddp \
  --samples_per_epoch 10000 \
  --use_pretrained_dimp 'False' \
  --workspace_dir $WORKSPACE \
  --lasot_rate 1 \
  --total_epochs 60 \
  --batch_size 4 \
  --num_workers 4 \
  --devices_id 0 1 2 3 4 5 6 7  # used gpus
