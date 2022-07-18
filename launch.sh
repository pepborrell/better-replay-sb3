#!/bin/bash

# Loading the required modules pre launch
source /etc/profile
module load anaconda/2022a
source /etc/profile.d/modules.sh
source $HOME/.bashrc
module load cuda/11.6
module load anaconda/2022a
module load mpi/openmpi-4.1.3
source $HOME/proxy.sh

export LC_CTYPE=en_US.UTF-8 LANG=en_US.UTF-8 LANGUAGE=en_US
export LD_LIBRARY_PATH=$HOME/.mujoco/mujoco210/bin:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/nvidia
export MJLIB_PATH=$HOME/.mujoco/mujoco-2.1.1/lib/libmujoco.so
export PYTHONPATH=$PYTHONPATH:/state/partition1/user/$USER/mujoco-py
export MUJOCO_GL=egl
export WANDB_MODE=online
export C_INCLUDE_PATH=$C_INCLUDE_PATH:$HOME/vendor/glew/include LD_PRELOAD=$HOME/vendor/glew/lib/libGLEW.so.2.2.0:/usr/lib/x86_64-linux-gnu/libEGL.so

########################## MINIGRID ###############################
sbatch launch_bash_script_gpu.sh scripts/train_dqn_minigrid.sh SimpleCrossingS9N1 uer;
# sbatch launch_bash_script_gpu.sh scripts/train_dqn_minigrid.sh SimpleCrossingS9N1 usr;
# sbatch launch_bash_script_gpu.sh scripts/train_dqn_minigrid.sh SimpleCrossingS9N1 usar;
# sbatch launch_bash_script_gpu.sh scripts/train_dqn_minigrid.sh SimpleCrossingS9N1 rusr;
# sbatch launch_bash_script_gpu.sh scripts/train_dqn_minigrid.sh SimpleCrossingS9N1 rusar;
