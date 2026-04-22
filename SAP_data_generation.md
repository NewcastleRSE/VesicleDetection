# Generating labelled data for the Synaptic Activation Prediction model

One aim of this project is to attain a model that can detect synaptic activation from the surrounding structures, size and shape of a bouton. 
This is utilising the number of PC+ (activated) vesicles in the synapse as a label for activation. 

The pipeline for data generation:

1. Run the base vesicle detection model prediction over your training data file. To do this, run `python apply.py <raw file> <trained model checkpoint> n` where the raw file expects a zarr, and the model checkpoint should be taken from a previous successful training run. ’n’ is a flag to prevent visualisation of results, this can be set to ‘y’ instead to output a visualisation.
_On Comet this is run through comet_predict.sh. Modify this script to change the input parameters as you desire, then submit to the queue with `sbatch comet_predict.sh`._
The results should appear within the raw zarr file, under <raw_zarr>/predict/Predictions/<date>/Hough_transformed and <raw_zarr>/predict/Predictions/<date>/candidates.csv. The Hough_transformed data fills in voxels for identified vesicles, with a label for PC+ or PC-, while the csv file contains x,y,z coordinates for the central point of identified vesicles with a score for how robust the prediction is, and a label for PC+ (1) or PC- (2).
2. Run the results through a clustering algorithm DBscan. 
This can use either the Hough_transformed output or the candidates.csv. We have found the best parameter set for the Hough_transformed (voxel based) so far to be eps: 6, ms: 60, while for the candidates.csv (vesicle centroid based) eps: 15, ms: 10 seems to be more effective.
From the base directory of the repository, run:
`python src/clustering/cluster_vesicles [prediction_path] [clusters_path] [eps] [min_samples]`
where the prediction path is the path to the zarr predict folder produced by step 1. The clusters_path is the save file name. 
If you wish to run for multiple parameters as a scan to find what works best, there is a shell script `cluster_loop.sh` which can be edited for the range of parameters to try, and then run from the clustering directory with `./cluster_loop.sh`.
_On Comet this can be run through the batch script cna_cluster.sh, open to modify the input parameters then submit with `sbatch cna_cluster.sh`.
3. The clustering will produce an `.npz` file. This can then be used to create a masked zarr file. To do this run:
`python src/clustering/parallel_masking.py <raw_zarr_path> <output_zarr_path> <clusters_path> --n_jobs 8 —plot`
The raw zarr path is for the original zarr file used for training (‘raw’ subfolder), the output is the save filename. Recommend putting this in the same zarr container as the original but named ‘mask’ or similar. The clusters path is for the nspz produced in step 2. —plot is an optional flag for plotting and —n_jobs lets you specify the parallel cores. 
If you want to mask just the vesicles rather than the full clusters (which tends to black out the whole bouton) then pass the hough_transformed path instead of the npz clusters file. 
_On Comet this is run via `cna_mask.sh`_
4. Once you are happy with the mask, this can then be chopped into crops with the cluster centre as the centrepoint of the crop, in order to produce a set of training data. 
Run this with: `python src/clustering/crop_clusters_parallel.py --raw_path <zarr_predict> --masked_path <zarr_masked> --npz <clustering_npz> --out_dir <output_filename> --n_jobs 8`
_On comet run this via `cna_cropclusters.sh`_
5. Then you need to create the labels. This can be done with `python get_cluster_counts.py <clusters.npz> <prediction.zarr> <output.csv>` This will create a csv file with the name and save location specified in the output.csv argument. This will contain cluster ids, positive counts.