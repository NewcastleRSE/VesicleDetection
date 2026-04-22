#!/bin/bash
#SBATCH --account=comet_rse_cna
#SBATCH --partition=gpu-s_free
#SBATCH --job-name=cna_vesicle_prediction
#SBATCH --mem=100G
#SBATCH --cpus-per-task=8
#SBATCH --time=05:00:00

date
#python --version
#source /nobackup/proj/comettestgroup1/frances_test/VesicleDetection/.venv/bin/activate
python --version
python /nobackup/proj/comet_rse_cna/VesicleDetection/src/clustering/parallel_masking.py '/nobackup/proj/comet_rse_cna/VesicleDetection/data/19-13_x1020-1728_y0-1024_z835-1670crop.zarr/predict/' '/nobackup/proj/comet_rse_cna/VesicleDetection/data/19-13_x1020-1728_y0-1024_z835-1670crop.zarr/predict/Mask_eps6_ms60' '/nobackup/proj/comet_rse_cna/VesicleDetection/data/clusters_19-13_x1020-1728_y0-1024_z835-1670_eps6ms60.npz' --n_jobs 8
date

