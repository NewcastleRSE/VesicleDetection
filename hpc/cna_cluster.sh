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
python /nobackup/proj/comet_rse_cna/VesicleDetection/src/clustering/cluster_vesicles.py '/nobackup/proj/comet_rse_cna/VesicleDetection/data/19-13_x1020-1728_y1020-2868_z835-1670crop.zarr/predict/Predictions/13_01_2026/Hough_transformed' '/nobackup/proj/comet_rse_cna/VesicleDetection/data/clusters_19-13_x1020-1728_y1020-2868_z835-1670_eps6ms60.npz' 6 60
date

