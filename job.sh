#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=6
#SBATCH --time=03-23
#SBATCH --partition=shas      
#SBATCH --qos=long
#SBATCH --output=sample-%j.out

module load python/3.6.5

python main_cgan.py
