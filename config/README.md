# Confifuration File Summary

This file provides details on each config file, with a brief description of what each parameter is used for. 

## Training Config

<details>
<summary>Click to see training configs</summary>

  ### augmentations
  > These are the [Gunpowder augmentations](https://funkelab.github.io/gunpowder/api.html#augmentation-nodes) used during training. They apply augmentations to both the raw and ground truth data, in order to produce new (fictious) training images to improve training diversity. 
  
  ### batch_size
  > The number of training image batches that should be run before updating the model.
  
  ### best_score_name
  > The score name used to display the best prediction from the model.
  > 
  > Options: `recall_1`, `recall_2`, `recall_average`, `precision_1`, `precision_2`, `precision_average`, `fscore_1`, `fscore_2`, `fscore_average`. 
  
  ### checkpoint_path
  > Redunant. Previously used to provide path to pretrained model. 
  
  ### clahe 
  > Boolean that determines whether the raw data should be clahe (Contrast Limited Adaptive Histogram Equalization) format. 
  
  ### has_mask
  > Boolean that states whether the training data contains a mask which restricts the area of the image that can be used for training (see [RandomLoication](https://funkelab.github.io/gunpowder/api.html#randomlocation) in Gunpowder documentation).
  
  ### input_shape
  > The shape (in voxel number) of the image used for training. This shape must be compatible with the UNet parameters, and be checked via the `check_output_shape.py` file. 
  
  ### iterations
  > The number of training iterations for the model. 
  
  ### save_every
  >The number of iterations between saving the model's checkpoints. By default these are saved in the working directory, but a location can be chosen by using the `checkpoint_basename` in `gp.torch.Train` in the `training.py` file. 
  
  ### snapshot_every
  > Not used currently. Leave value at 0. 
  
  ### val_every
  > The number of iterations between running a validation run.

</details>

## Model Config

<details>
<summary>Click to see model configs</summary>

  ### constant_upsample
  > Boolean that determines whether to use constant upsampling in the UNet.
  
  ### downsample_factors
  > The factors used to downsample and upsample in the UNet.The first sets the downsampling factor in $(z,y,x)$ between layers of the UNet. The second sets the upsampling factor in $(z,y,x)$. It is reccommended to use the same downsampling and upsampling factors.
  > 
  > The tuples should reflect the isotropic nature of the data (e.g. (2,2,2) for isotropic vs (1,2,2) for anisotropic).
  
  ### fmap_inc_factors
  > The multiplicative factor for feature maps between layers.
  >
  > If layer $n$ has $k$ feature maps, layer $(n+1)$ will have $(k *$ fmap_inc_factor $)$ feature maps.
  
  ### fmaps
  > The number of feature maps in the first layer of the UNet.
  
  ### padding
  > How to pad convolutions within the UNet.
  >
  > Options: `same` or `valid`.

</details>

## Post Processing Config

<details>
<summary>Click to see post processing configs</summary>

  ### combine_pos_neg
  > Boolean that determines whether to combine the PC+ and PC- probability distributions before determining candidate vesicle.
  >
  > If set to true, this will result in a post processing procedure that favours finding vesicle existance and later labelling, rather than looking for PC+ and PC- vesicles independently.
  
  ### maxima_threshold 
  > The threshold used to determine whether a candidate vesicle is accepted, i.e. a threshold for the confidence score of the model.

</details>

## Tiff to Zarr Train Config

> [!NOTE]
> The `tiff_to_zarr_train.py` file was created when the groundtruth was provided in separate tiff files (i.e. PC+ labels and PC- labels were separate tiff files). As such it expects the files in this format, which are provided by this config file. This needs to be updated to account for the new way data is saved using the Napari labelling procedure. 

<details>
  <summary>Click to see tiff to train zarr configs</summary>

  ### attributes
  > The attributes of the zarr data to be saved.

  ### output_zarr_path
  > The path to the directory in which to save the zarr data.

  ### path_to_gt_PC_neg_tiff_train
  > The path to the tiff file containing the groundtruth labels for the PC- vesicles used for training.

  ### path_to_gt_PC_neg_tiff_validate
  > The path to the tiff file containing the groundtruth labels for the PC- vesicles used for validation.

  ### path_to_gt_PC_pos_tiff_train
  > The path to the tiff file containing the groundtruth labels for the PC+ vesicles used for training.

  ### path_to_gt_PC_pos_tiff_validate
  > The path to the tiff file containing the groundtruth labels for the PC+ vesicles used for validation.

  ### path_to_raw_tiff_train
  > The path to the tiff file containing the raw training data.

  ### path_to_raw_tiff_validate
  > The path to the tiff file containing the raw validation data. 
  
</details>

## Tiff to Zarr Predict Config 

<details>
  <summary>Click to see tiff to zarr predict configs</summary>

  ### attributes
  > The attributes of the zarr data to be saved.

  ### output_zarr_path 
  > The path to the directory in which to save the zarr data.

  ### path_to_raw_tiff
  > The path to the tiff file containing the raw data. 
  
</details>
