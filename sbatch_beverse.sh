#!/usr/bin/env bash
#SBATCH --job-name=beverse_train
#SBATCH --output=beverse_train%j.log
#SBATCH --error=beverse_train%j.err
#SBATCH --mail-user=kraussn@uni-hildesheim.de
#SBATCH --partition=STUD
#SBATCH --gres=gpu:1
set -e

source /home/kraussn/switch-cuda.sh 11.3

source /home/kraussn/anaconda3/bin/activate /home/kraussn/anaconda3/envs/motion_detr

cd /home/kraussn/EMT_BEV/  # navigate to the directory if necessary


srun /home/kraussn/anaconda3/envs/motion_detr/bin/python3 setup.py develop
srun /home/kraussn/anaconda3/envs/motion_detr/bin/python3 home/kraussn/EMT_BEV/projects/mmdet3d_plugin/models/ops/setup.py build install
srun /home/kraussn/anaconda3/envs/motion_detr/bin/python3 /home/kraussn/EMT_BEV/tools/create_data.py nuscenes --root-path /home/kraussn/EMT_BEV/data/nuscenes --out-dir /home/kraussn/EMT_BEV/data/nuscenes --extra-tag nuscenes
# srun /home/kraussn/anaconda3/envs/motion_detr/bin/python3 train_debugg_cluster.py 
# home/kraussn/EMT_BEV/projects/mmdet3d_plugin/models/ops/setup.py build install