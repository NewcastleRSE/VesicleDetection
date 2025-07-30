#!/bin/bash
#SBATCH --partition=default_free
#SBATCH --mem=1000M
#SBATCH --cpus-per-task=2
#SBATCH --time=05:00:00

date
#module load Python/3.13.1-GCCcore-14.2.0
#module load pytorch-env
#module load Cython
#module load numpy
#module list
python --version
source /nobackup/proj/comettestgroup1/frances_test/VesicleDetection/.venv/bin/activate
python --version
python /nobackup/proj/comettestgroup1/frances_test/VesicleDetection/apply.py '/nobackup/proj/comettestgroup1/frances_test/VesicleDetection/data/19-13_subvolume_0647-1670_6x6x6nm.zarr' '/nobackup/proj/comettestgroup1/frances_test/VesicleDetection/saved_models/training02/model_checkpoints/fscore_average' y
date

