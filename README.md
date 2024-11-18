# Automated Vesicle Detection

## 🌟 Highlights

- Customisable machine learning model
- Automate detection of PC+ and PC- vesicles in EM data
- Resulting labels can be used as ground truth to train further models

## ℹ️ Overview

This code is designed to eliviate the requirement of hand labelling PC+ and PC- vesicles within photoconverted EM data, by employing machine learning learning methods. This not only frees us researchers time, but allows for a more consistent labelling procedure. The programme assumes perfectly spherical vesicles, and adds them to the image in order of confidence, while ensuring no physical issues (e.g. no two vesicles can overlap). The results can be visualised using Napari, allowing for an effective workflow.

The programme is customisable via configuration files, and the user can train their own model specifically for their data. This allows for optimal performance across different experimental setups that result in different data specifications (e.g. resolution). 

## ⬇️ Installation 

This code comes equipped with `setup.py` and `requirements.txt` files to allow for simple creation of a conda virtual enviroment to run the code from. After sucessfully downloading all files to your computer, the user should take the following steps to get the code functional. 
- Ensure conda is installed on the system.
- Navigate to the downloaded directory containing the code (either in terminal or in a IDE such as VS Code).
- Run the following command
  ```bash
  conda create -p venv python==3.12.4 -y
  ```
- Activate the conda environment as instructed
- Install the required packaged by running the following command
  ``` bash
  pip install -r requirements.txt
  ```

This should create a virtual conda enviroment, with name `venv`, within the directory containing the code. 

> [!WARNING]
> Issues may occur with the above installation due to the `cython` and `funlib-evaluate` packages. If an error is raised saying it cannot find cython, it is advised to temporarily delete the funlib line from the requirements.txt file, run the pip install command to ensure cython is installed, and then paste the funlib line back in. Running the pip install command a second time should now finish the installation of all required packages.

> [!IMPORTANT]
>  This code has been set up on a Linux Ubuntu machine, and the packages are known to work on this OS. Issues may occur with other systems, and installation of packages may need to be trouble shooted. There are plans in the future to impliment a docker deployment of this code, and these installation instructions will be updated when this is completed. 

## 📁 Requirements of saved data

The programme requires the data to be saved in a particular manner. All data is required to be zarr data (with accompanying `.zgroup`, `.zarray` and `.zattrs` files) with the structure listed below. The minimum requirements for attributes are `axes`, `offset` and `resolution`. 

Training:

```bash
data_for_training.zarr
    ├── train
    │      ├── raw
    │      │    ├── 0.0.0
    │      │    ├── ...
    │      │    ├── .zarray
    │      │    └── .zattrs
    │      ├── raw_clahe (optional)
    │      │    ├── 0.0.0
    │      │    ├── ...
    │      │    ├── .zarray
    │      │    └── .zattrs
    │      ├── gt
    │      │    ├── 0.0.0
    │      │    ├── ...
    │      │    ├── .zarray
    │      │    └── .zattrs
    │      └── .zgroup
    └── validate
           ├── raw
           │    ├── 0.0.0
           │    ├── ...
           │    ├── .zarray
           │    └── .zattrs
           ├── raw_clahe (optional)
           │    ├── 0.0.0
           │    ├── ...
           │    ├── .zarray
           │    └── .zattrs
           └── gt
               ├── 0.0.0
               ├── ...
               ├── .zarray
               └── .zattrs
```
Prediction (the contents of this can be contained in the training directionary, if wanted):

```bash
data_for_prediction.zarr
      └── predict
            ├── raw
            │    ├── 0.0.0
            │    ├── ...
            │    ├── .zarray
            │    └── .zattrs
            └── raw_clahe (optional)
                 ├── 0.0.0
                 ├── ...
                 ├── .zarray
                 └── .zattrs
```

## 🚀 Predicting On Data 

In order to predict on new EM data, the user should run the `apply.py` file and follow the instructions. The data path should point to a zarr group with the following structure:

```bash
data.zarr
  └── predict
        ├── raw
        └── raw_clahe (optional)
```

The path to the model checkpoint should lead directly to the model checkpoint file. That is, if using the in built saving system, the user would type `saved_models/dd_mm_yyyy/model_checkpoints/score_name`. 

The prediction result will then be saved within the provided zarr group. By default it saves as follows:
```bash
data.zarr
  └── predict
        ├── raw
        ├── raw_clahe (optional)
        └── Predictions
              └── dd_mm_yyy
                    └── Hough_transformed
```

## 🧰 Model Outline

Pytorch is used to build the model, which consists of a UNet along with a convolution final layer which returns the probabilites of a voxel belonging to one of three classes: (a) background (b) PC+ or (c) PC-. These probabilites are then post processed (using a custom version of a Hough Transform) in order to search for spherically symmetric regions of high probabilites. The result is then a collection of spheres of set size which label the PC+ and PC- vesicles. This prediction can be overlayed over the original data in Napari, in order to inspect the prediction. 

The model makes use of a customer version of the cross entropy loss, which accounts for masked data. The weight for background contribution to the loss is reduced to 0.01 (compared to 1.0 for PC+ and PC-), to account for data imbalance. 

## 🛠️ Training A Network

The user can use the `run.py` file to train their own network from scratch, in order to optimise the model to their specific data. This training process requires the user to prepare a collection of training and validation data, in which the vesicles have been hand labelled. This data should then be saved as zarr data (which can be done using the provided `tiff_to_zarr_train.py` and `tiff_to_zarr_tain_config.yaml` files) as per the requirements above. The configurations should then be set using the `model_config.yaml`, `training_config.yaml` and `post_processing_config.yaml` files. 

> [!NOTE]
>  When training is initiated, the programme with create another zarr array, `target`, which is simply a copy of `gt` but with the data-type set altered to fit the code.

When training, each validation prediction is given a set of 9 scores: reacll, precision and fscore for PC+ and PC- separetly as well as an average for each. The model is saved according to these scores: the checkpoints that perform best for each score are individually saved, along with their validation results. Additionally, for each score, two csv files will be created: `candidates.csv` and `stats.csv`. The former stores the centre coordinates for each vesicle, its confidence score and its label (1 = P+, 2 = PC-). The latter saves the scores for that particular iteration, along with the loss. Three further files are also saved: `best_scores.yaml`, `summary.yaml` and `training_config_used.yaml`: the first simply stores the best scores achieved, along with their iteration number; the second provides a summary of the training run, including how many PC+ and PC- vesicles were predicted, based on the best score specified in the config file; the third simply copies across the training config file used.

It is possible to continue training a model, and this option is provided as a terminal input during the run. When done, the programme will first copy over all the saved model files from the previous run, and then resume training from there. An asterisk is used to indicate iteration numbers that belong to the previous training run. For example, assume the training runs are for 10000 iterations, then in the second run "Iteration 9600*" is the 9600th iteration of the first run, while "Iteration 1000" is the 1000th iteration in the second run (i.e. the 11000th total iteration). 

## 💭 Feedback & Contributing 

Any feedback and/or possible contributions to this code are welcome in the Discussions tab of this repo. Users are also welcome to open issues for bugs/feature requests. 
