#!/bin/bash
#SBATCH -A uppmax2020-2-2
#SBATCH -p core -n 8
#SBATCH -M snowy
#SBATCH -t 20:00:00 
#SBATCH -J transparent 
#SBATCH --gres=gpu:1


python Execution.py
