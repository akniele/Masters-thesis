#!/bin/bash
#SBATCH -A uppmax2020-2-2
#SBATCH -p core -n 4
#SBATCH -M snowy
#SBATCH -t 1:30:00 
#SBATCH -J baseline 
#SBATCH --gres=gpu:1

LR=$1  # initial learning rate
EPOCHS=$2  # max number of epochs

python Execution.py $LR $EPOCHS
