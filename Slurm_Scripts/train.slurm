#!/bin/bash
#SBATCH -p gpu
#SBATCH --gres=gpu:2
#SBATCH -C gmem32
#SBATCH -c 6
#SBATCH --mem=40000
#SBATCH -o exp23out
#SBATCH --job-name=exp23
#SBATCH --nodes=1
#SBATCH --time=48:00:00

module load anaconda3/2019.02
source activate fallenv
python3 ../train.py
