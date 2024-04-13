#!/bin/bash
#SBATCH --ntasks=1                     
#SBATCH --nodes=1 
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=4              
#SBATCH --mem-per-cpu=2G
#SBATCH --time=00-12:00:00 


# Go to the current folder
SCRIPTDIR=$(scontrol show job "$SLURM_JOB_ID" | awk -F= '/Command=/{print $2}')
SCRIPTDIR=$(realpath "$SCRIPTDIR")
cd $(dirname "$SCRIPTDIR")

# Setup environment (adapt if necessary)
source /srv/beegfs-benderdata/scratch/${USER}/data/conda/etc/profile.d/conda.sh
conda activate py39

# Add environ var
export DATA=/srv/beegfs-benderdata/scratch/ac_course/data/project_1/miniscapes.tar
export TMPDIR=/scratch/${USER}
export SAVEDIR=/srv/beegfs-benderdata/scratch/${USER}/data/ex1_submission

# setup weights and biases
export WANDB_API_KEY=$(cat "wandb.key")
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

# You can specify the hyperparameters and the experiment name here. # --name aspp_fft_3try_halfchann$
python -m source.scripts.train \
  --log_dir ${SAVEDIR} \
  --dataset_root ${TMPDIR}/miniscapes \
  --name multitask_test \
  \
  --tasks depth semseg \
  --loss_weight_semseg 0.5 \
  --loss_weight_depth 0.5 \
  \
  --model_name deeplabv3p_multitask \
  --pretrained true \
  \
  --optimizer adam \
  --optimizer_lr 0.0001 \
  --optimizer_weight_decay 0.001 \
  --lr_scheduler poly \
  --lr_scheduler_power 0.9 \
  \
  \
  --batch_size 16 \
  --num_epochs 50 \
  \
  --workers ${SLURM_CPUS_PER_TASK} \
  --workers_validation ${SLURM_CPUS_PER_TASK} \
  --batch_size_validation 16 \
  --optimizer_float_16 no
  # ... you can pass further arguments as specified in utils/config.py
  # DO NOT FORGET ADDING BACKSLASHES for additional flags (except the last one)

# END YOUR CHANGES HERE

