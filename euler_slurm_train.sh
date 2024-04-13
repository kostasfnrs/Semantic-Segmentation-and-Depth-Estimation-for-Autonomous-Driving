#!/bin/bash
#SBATCH --ntasks-per-node=4
#SBATCH --output=%J.out
#SBATCH --tmp=30000
#SBATCH --cpus-per-task=4            
#SBATCH --mem-per-cpu=4G
#SBATCH --time=23:00:00
#SBATCH --gpus=1






# Go to the current folder
SCRIPTDIR=$(scontrol show job "$SLURM_JOB_ID" | awk -F= '/Command=/{print $2}')
SCRIPTDIR=$(realpath "$SCRIPTDIR")
cd $(dirname "$SCRIPTDIR")

# Add environ var
export DATA=/cluster/scratch/${USER}/miniscapes.tar
export SAVEDIR=/cluster/scratch/${USER}/ex1_submission_$NAME


# euler modules
module load gcc/8.2.0
module load python_gpu/3.11.2
module load eth_proxy

# Nvidia check
nvidia-smi
nvidia-smi -q -x | grep gpu_name

# Install micromamba
cd ${TMPDIR}
wget https://micro.mamba.pm/api/micromamba/linux-64/latest
tar -xvf ./latest 
./bin/micromamba shell init -s bash -p ${TMPDIR}/environments
source ~/.bashrc
yes | ./bin/micromamba create -n cv_env python=3.9 -c conda-forge
micromamba activate cv_env
cd $(dirname "$SCRIPTDIR")
pip install -r ./requirements.txt 
rm -rf ~/.cache/pip


# setup weights and biases
export WANDB_API_KEY=$(cat "wandb_andreas.key")
export WANDB_DIR=${TMPDIR}
export WANDB_CACHE_DIR=${TMPDIR}
export WANDB_CONFIG_DIR=${TMPDIR}

# Extract dataset (do not change this)
echo "Extract miniscapes"
mkdir -p ${TMPDIR}
mkdir -p ${SAVEDIR}
tar -xf ${DATA} -C ${TMPDIR}

# BEGIN YOUR CHANGES HERE
export TEAM_ID=7

# Run training
echo "Start training"

# You can specify the hyperparameters and the experiment name here.
python -m source.scripts.train \
  --log_dir ${SAVEDIR} \
  --dataset_root ${TMPDIR}/miniscapes \
  --name adaptive_depth_with_posemb_expsin4_bins512 \
  \
  --tasks depth \
  --num_bins 400 \
  --num_heads 8 \
  --expansion 4 \
  \
  --model_name adaptive_depth \
  --pretrained true \
  \
  --optimizer adam \
  --optimizer_lr 0.0001 \
  --optimizer_weight_decay 0.001 \
  --lr_scheduler poly \
  --lr_scheduler_power 0.9 \
  \
  --accumulate_grad_batches 1 \
  \
  --batch_size 4 \
  --num_epochs 25 \
  \
  --workers ${SLURM_CPUS_PER_TASK} \
  --workers_validation ${SLURM_CPUS_PER_TASK} \
  --batch_size_validation 4 \
  --optimizer_float_16 no
  # ... you can pass further arguments as specified in utils/config.py
  # DO NOT FORGET ADDING BACKSLASHES for additional flags (except the last one)

# END YOUR CHANGES HERE

