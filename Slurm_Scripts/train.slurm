#!/bin/bash
#SBATCH --gres=gpu:2
#SBATCH -C gmem16
#SBATCH -c 6
#SBATCH --mem=90000
#SBATCH -o exp1out
#SBATCH --job-name=exp1
#SBATCH --nodes=1
#SBATCH --time=10:00:00

module load anaconda/anaconda3
source activate fallenv
python3 ../train.py
