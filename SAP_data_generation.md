# Generating labelled data for the Synaptic Activation Prediction model

One aim of this project is to attain a model that can detect synaptic activation from the surrounding structures, size and shape of a bouton. 
This is utilising the number of PC+ (activated) vesicles in the synapse as a label for activation. 

The pipeline for data generation:


## Pre-processing

#### Generating a new large crop for the data pipeline 
Large EM images may need to be made more manageable by cropping into smaller segments to run through the pipeline, or may have areas with large artefacts that are better ignored. The function `src/processing/pre_processing/crop.py` Lets you crop an image by passing it the image name and cropping ranges, and optionally the output filename. 
`python src/processing/pre_processing/crop.py “path_to_crop” -r “[[xmin, xmax],[ymin,ymax],[zmin,zmax]]”`  

#### Converting tiff to zarr
The EM images generally come in tiff format, but the pipeline is expecting a .zarr format. There are two different functions for converting the tif to a zarr, depending on whether you intend the image to be used for training the Vesicle detection model, or to be used with a pre-trained model to generate a prediction for feeding through to the Synaptic Activation Prediction model.
To use the conversion functions, you need to fill in the config files in `config/tiff_to_zarr_predict_config.yaml` or `config/tiff_to_zarr_train_config.yaml` with the paths to the file to be converted, and the output filename ass well as setting any offset and resolution. Once the config files have been set, the image can be processed by running either `python src/processing/pre_processing/tiff_to_zarr_predit.py` or `python src/processing/pre_processing/tiff_to_zarr_predit.py` 
The outputted zarr should then be ready to pass to the data pipeline for the synaptic activation detection, or to the vesicle detection model training depending on whether you used predict or train.

## Generating labelled data for synaptic activation detection

One aim of this project is to attain a model that can detect synaptic activation from the surrounding structures, size and shape of a bouton. 
This is utilising the number of PC+ (activated) vesicles in the synapse as a label for activation. 

The pipeline for data generation:

1. Run the base vesicle detection model prediction over your training data file. To do this, run `python apply.py <raw file> <trained model checkpoint> n` where the raw file expects a zarr, and the model checkpoint should be taken from a previous successful training run. ’n’ is a flag to prevent visualisation of results, this can be set to ‘y’ instead to output a visualisation.
_On Comet this is run through comet_predict.sh. Modify this script to change the input parameters as you desire, then submit to the queue with `sbatch comet_predict.sh`._
The results should appear within the raw zarr file, under <raw_zarr>/predict/Predictions/<date>/Hough_transformed and <raw_zarr>/predict/Predictions/<date>/candidates.csv. The Hough_transformed data fills in voxels for identified vesicles, with a label for PC+ or PC-, while the csv file contains x,y,z coordinates for the central point of identified vesicles with a score for how robust the prediction is, and a label for PC+ (1) or PC- (2).


2. Run `cluster_crop_pipeline.py` in order to generate the datasets needed for the Synaptic Activation Prediction model. This pipeline runs all of the below steps, and will visualise the clusters and check if you are happy before proceeding. 
Example use:
```python

python src/clustering/cluster_crop_pipeline.py \
    --predictions_path /<path_to_project>/CorrelatingNeuronalActivity/VesicleDetection/data/<volume_id>crop.zarr/predict/Predictions/<date> \
    --eps 15 \
    --min_samples 10 \
    --raw_path /<path_to_project>/CorrelatingNeuronalActivity/VesicleDetection/data/<volume_id>crop.zarr/predict/ \
    --dilation 2 \
    --n_jobs 8 \
    --chunk_size 100000 \
    --min_size  200\
    --max_size 10000 \
    --use_filenames

```
Where the parameters can be chosen and the parts in ‘<>’ need to be filled in.

2a. Run the results through a clustering algorithm DBscan. 
This can use either the Hough_transformed output or the candidates.csv. We have found the best parameter set for the Hough_transformed (voxel based) so far to be eps: 6, ms: 60, while for the candidates.csv (vesicle centroid based) eps: 15, ms: 10 seems to be more effective.

From the base directory of the repository, run:
`python src/clustering/cluster_vesicles [prediction_path] [clusters_path] [eps] [min_samples]`
where the prediction path is the path to the zarr predict folder produced by step 1 if you want to cluster using the Hough transformed voxels, OR the candidates csv file which should be in the same folder as the Hough transformed data. The clusters_path is the save file name. 

If you wish to run for multiple parameters as a scan to find what works best, there is a shell script `cluster_loop.sh` which can be edited for the range of parameters to try, and then run from the clustering directory with `./cluster_loop.sh`.
_On Comet this can be run through the batch script cna_cluster.sh, open to modify the input parameters then submit with `sbatch cna_cluster.sh`.


2b. The clustering will produce an `.npz` file. This can then be used to create a masked zarr file. To do this run:
`python src/clustering/parallel_masking.py <raw_zarr_path> <output_zarr_path> <clusters_path> --n_jobs 8 —plot`
The raw zarr path is for the original zarr file used for training (‘raw’ subfolder), the output is the save filename. Recommend putting this in the same zarr container as the original but named ‘mask’ or similar. The clusters path is for the npz file produced in step 2a. —plot is an optional flag for plotting and —n_jobs lets you specify the parallel cores. 
If you want to mask just the vesicles rather than the full clusters (which tends to black out the whole bouton) then pass the hough_transformed path instead of the npz clusters file. 
_On Comet this is run via `cna_mask.sh`_


2c. Once you are happy with the mask, this can then be chopped into crops with the cluster centre as the centrepoint of the crop, in order to produce a set of training data. 
Run this with: `python src/clustering/crop_clusters_parallel.py --raw_path <zarr_predict> --masked_path <zarr_masked> --npz <clustering_npz> --out_dir <output_filename> --n_jobs 8`
_On comet run this via `cna_cropclusters.sh`_


2d. Then you need to create the labels. This can be done with `python get_cluster_counts.py <clusters.npz> <prediction.zarr> <output.csv>` This will create a csv file with the name and save location specified in the output.csv argument. This will contain cluster ids, positive counts.


3. Once you are ready to generate the final labels for the crops, whether you have manually labelled them or decided to skip that step, you can generate the final labels by running: `python src/clustering/combine_counts_and_crops.py <folder_name>` where the folder name should be the folder generated by the pipeline, containing a subfolder with a name ending in `cropped_convexhull` containing a subfolder `masked` containing the .h5 masked crops, and also a subfolder ending in `cropped_vesicle` again containing a subfolder `masked` with the h5 files for the vesicle masked crops. The folder should also contain a `*labels.csv` file with the labels per cluster generated by the last step in part 2, or automatically generated by the pipeline. 
So the directory structure expected is:
```
<folder_name>
    |
    > <folder_name>_cropped_convexhull
        |
        > masked
            |
            + cluster_<id>_masked.h5
            +         …
        |
        > raw
    |
    > <folder_name>_cropped_vesicle
        |
        > masked
            |
            + cluster_<id>_masked.h5
            +         …
        |
        > raw
    |
    + <folder_name_labels.csv

```

If you want to combine multiple crops into one, just have all of them in the same folder, e.g.:
```
<crops_to_combine_folder>
    |
    > <prefix_1>_cropped_convexhull
        |
        > masked
            |
            + cluster_<id>_masked.h5
            +         …
        |
        > raw
    |
    > <prefix_1>_cropped_vesicle
        |
        > masked
            |
            + cluster_<id>_masked.h5
            +         …
        |
        > raw
    |
    + <prefix_1>_labels.csv
    |
    > <prefix_2>_cropped_convexhull
        |
        > masked
            |
            + cluster_<id>_masked.h5
            +         …
        |
        > raw
    |
    > <prefix_2>_cropped_vesicle
        |
        > masked
            |
            + cluster_<id>_masked.h5
            +         …
        |
        > raw
    |
    + <prefix_2>_labels.csv

   …


```

This final step should combine the crops into one folder if there are multiple prefixes, and if not just rename the crops in their current folder to have the appropriate prefix, and create an output `<prefix>_labelscombined.csv` which will also combine cluster labels where there is a `*ratings.csv` file to be processed with any clusters to combine. 

The `<prefix>_labelscombined.csv` and the folder with the crops can then be shared for model training.
                      