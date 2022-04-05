#!/bin/bash
#SBATCH -p gpu
#SBATCH --gres=gpu:2
#SBATCH -C gmem24
#SBATCH -c 6
#SBATCH --mem=40000
#SBATCH -o exp13out
#SBATCH --job-name=exp13
#SBATCH --nodes=1
#SBATCH --time=48:00:00

module load anaconda3/2019.02
source activate fallenv
python3 ../train.py
