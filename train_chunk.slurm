#!/bin/bash -l
#
#SBATCH --gres=gpu:a40:1
#SBATCH --time=6:00:00
#SBATCH --export=NONE

unset SLURM_EXPORT_ENV

module load python
module load cuda/11.8
conda activate surffill


SOURCE_DIR=$1
OUTPUT_DIR=$2
CHUNK_NUM=$3
BLOCK_X=$4
BLOCK_Y=$5
BLOCK_Z=$6
BLOCK_L=$7
BLOCK_W=$8
BLOCK_H=$9

echo "Source Directory: ${SOURCE_DIR}"
echo "Output Directory: ${OUTPUT_DIR}"
echo "Chunk Number: ${CHUNK_NUM}"
echo "Block Dimensions: X=${BLOCK_X}, Y=${BLOCK_Y}, Z=${BLOCK_Z}, L=${BLOCK_L}, W=${BLOCK_W}, H=${BLOCK_H}"



# Train the chunk
python train_single.py -s ${SOURCE_DIR} --iterationS ${CHUNK_NUM} --unique_str_all ${OUTPUT_DIR} --block ${BLOCK_X} ${BLOCK_Y} ${BLOCK_Z} ${BLOCK_L} ${BLOCK_W} ${BLOCK_H}


echo "CHUNK " ${CHUNK_NUM} " FULLY TRAINED."
