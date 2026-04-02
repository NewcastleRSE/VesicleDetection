#!/bin/bash
#SBATCH --partition=default_free
#SBATCH --mem=100G
#SBATCH --cpus-per-task=2
#SBATCH --time=10:00:00

date
source /nobackup/proj/comettestgroup1/frances_test/VesicleDetection/.venv/bin/activate
python /nobackup/proj/comettestgroup1/frances_test/VesicleDetection/src/cluster_vesicles.py '/nobackup/proj/comettestgroup1/frances_test/VesicleDetection/data/19-13_subvolume_0647-1670_6x6x6nm.zarr/predict/Predictions/31_07_2025/Hough_transformed'
date

