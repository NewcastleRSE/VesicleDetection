# Confifuration File Summary

## Training Config

### augmentations
> These are the [Gunpowder augmentations](https://funkelab.github.io/gunpowder/api.html#augmentation-nodes) used during training. They apply augmentations to both the raw and ground truth data, in order to produce new (fictious) training images to improve training diversity. 

### batch_size
> The number of training image batches that should be run before updating the model.

### best_score_name
> The score name used to display the best prediction from the model. 

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
