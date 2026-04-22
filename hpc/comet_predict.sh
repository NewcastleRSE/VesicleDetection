#!/bin/bash
#SBATCH --account=comet_rse_cna
#SBATCH --partition=gpu-s_free
#SBATCH --job-name=cna_vesicle_prediction
#SBATCH --mem=100G
#SBATCH --cpus-per-task=5
#SBATCH --time=05:00:00

date
#python --version
#source /nobackup/proj/comettestgroup1/frances_test/VesicleDetection/.venv/bin/activate
python --version
python /nobackup/proj/comet_rse_cna/VesicleDetection/apply.py '/nobackup/proj/comet_rse_cna/VesicleDetection/data/19-13_x1020-1728_y0-1024_z0-835crop.zarr' '/nobackup/proj/comet_rse_cna/VesicleDetection/Model_checkpoints/Saved_models/training02/model_checkpoints/fscore_average' n
date

