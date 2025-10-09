#!/bin/bash

# Ranges for parameters
eps_values=(6)
min_samples_values=(200)

# Input data path
data_path="/Users/administrator/Documents/CorrelatingNeuronalActivity/VesicleDetection/data/19-13_subvolume_0647-1670_6x6x6nm.zarr/predict/Predictions/31_07_2025/Hough_transformed"
output_file="/Users/administrator/Documents/CorrelatingNeuronalActivity/VesicleDetection/data/clusters/clusters_1913sbv_"

# Loop through all combinations
for eps in "${eps_values[@]}"; do
    for min_samples in "${min_samples_values[@]}"; do
        echo "Running with eps=$eps, min_samples=$min_samples"
        python cluster_vesicles.py "$data_path" "$output_file" "$eps" "$min_samples"
    done
done
