#!/bin/bash
#SBATCH --partition=default_free
#SBATCH --mem=100G
#SBATCH --cpus-per-task=2
#SBATCH --time=10:00:00

date
python --version
source /nobackup/proj/comettestgroup1/frances_test/VesicleDetection/.venv/bin/activate
python --version
python /nobackup/proj/comettestgroup1/frances_test/VesicleDetection/apply.py '/nobackup/proj/comettestgroup1/frances_test/VesicleDetection/data/19-13_subvolume_0647-1670_6x6x6nm.zarr' '/nobackup/proj/comettestgroup1/frances_test/VesicleDetection/saved_models/training02/model_checkpoints/fscore_average' n
date

