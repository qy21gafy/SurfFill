#!/bin/bash -l
#
#SBATCH --gres=gpu:a40:1
#SBATCH --time=6:00:00
#SBATCH --export=NONE

unset SLURM_EXPORT_ENV

module load python
module load cuda/11.8
conda activate surffill

# Train the chunk
python train_parallel.py -s path

