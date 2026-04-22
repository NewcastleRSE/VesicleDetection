#!/bin/bash
#SBATCH --account=comet_training
#SBATCH --partition=short_free
#SBATCH --job-name=cna_vesicle_prediction
#SBATCH --ntasks=1    # CPU cores requested
#SBATCH --time=08:00  # HH:MM
#SBATCH --nodes=1     # number of nodes
#SBATCH --mail-user=nfh29@ncl.ac.uk

python apply.py

