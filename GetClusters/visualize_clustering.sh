#!/bin/bash
#SBATCH -A uppmax2020-2-2
#SBATCH -p core -n 4
#SBATCH -M snowy
#SBATCH -t 00:20:00 
#SBATCH -J transparent 
#SBATCH --gres=gpu:1


python visualize_clustering.py
