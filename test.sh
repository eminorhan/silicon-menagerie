#!/bin/bash

#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=200GB
#SBATCH --time=00:10:00
#SBATCH --job-name=test_zoo
#SBATCH --output=test_zoo_%A_%a.out
#SBATCH --array=0

module purge
module load cuda/11.3.1

python -u /scratch/eo41/saycam-zoo/test.py \

echo "Done"