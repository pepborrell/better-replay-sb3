#!/bin/bash

# Slurm sbatch options
#SBATCH -o slurm_logs/scriptGPU.sh.log-%j
#SBATCH -c 20
#SBATCH --gres=gpu:volta:1

EXTRA_ARGS=${@:1}

# Config if using mujoco
# echo "copying mujoco-py";
# mkdir -p /state/partition1/user/$USER;
# cp -r /home/gridsan/$USER/mujoco-py /state/partition1/user/$USER/;
# echo "finished";
# export MUJOCO_GL=egl;

export WANDB_MODE=online
wandb online
bash $EXTRA_ARGS --seed 100 & sleep 30;
bash $EXTRA_ARGS --seed 200 & sleep 30;
bash $EXTRA_ARGS --seed 300;
