#!/bin/bash
#SBATCH --ntasks-per-node=4
#SBATCH --output=%J.out
#SBATCH --tmp=30000
#SBATCH --cpus-per-task=4            
#SBATCH --mem-per-cpu=4G
#SBATCH --time=48:00:00
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

# You can specify the hyperparameters and the experiment name here. # --name aspp_fft_3try_halfchannels \
python -m source.scripts.train \
  --log_dir ${SAVEDIR} \
  --dataset_root ${TMPDIR}/miniscapes \
  --name task5_new_decoder_resnet34_32bs_350epochs \
  \
  --tasks semseg depth \
  --loss_weight_semseg 0.5 \
  --loss_weight_depth 0.5 \
  \
  --aug_geom_tilt_max_deg 0.0 \
  --aug_geom_reflect False \
  --aug_geom_wiggle_max_ratio 0.0 \
  --aug_geom_scale_max 1.0 \
  --aug_geom_scale_min 1.0 \
  \
  --model_name deeplabv3p_multitask_task5 \
  --model_encoder_name resnet34 \
  --pretrained true \
  \
  --optimizer adam \
  --optimizer_lr 0.0001 \
  --optimizer_weight_decay 0.001 \
  --lr_scheduler poly \
  --lr_scheduler_power 0.9 \
  \
  --accumulate_grad_batches 2 \
  \
  --batch_size 8 \
  --num_epochs 350 \
  \
  --workers ${SLURM_CPUS_PER_TASK} \
  --workers_validation ${SLURM_CPUS_PER_TASK} \
  --batch_size_validation 8 \
  --optimizer_float_16 no
  # ... you can pass further arguments as specified in utils/config.py
  # DO NOT FORGET ADDING BACKSLASHES for additional flags (except the last one)

# END YOUR CHANGES HERE

