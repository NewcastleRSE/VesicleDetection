
import os
from datetime import datetime
import numpy as np
import skimage.io
import zarr
import torch

from src.data_loader import EMData
from src.processing.predict import Prediction
from src.model.model import DetectionModel
from src.processing.post_processing.hough_detector import HoughDetector
from src.directory_organisor import create_unique_directory_file
from src.visualisation import imshow_napari_prediction

from config.load_configs import TRAINING_CONFIG
from config.load_configs import POST_PROCESSING_CONFIG


label_background = 0
label_pos = 1
label_neg = 2


def Apply(
        zarr_path: str, model_checkpoint: str,
        save_all_labels_in_one_tiff_file: bool = False, tiff_file_name_of_all_labels: None | str = None,
        save_different_labels_in_different_tiff_files: bool = False,
        tiff_file_names_of_different_labels: None | str = None):

    """
        Use a pretrained vesicle detection model to predict vesicles in unlablled data. 

        Parameters 
        -------------------
        zarr_path (str):
            Path to the zarr group that contains the 'predict' zarr group within it. This 
            path will be fed into the EMData class. 
        model_checkpoint (str):
            Path to the model that should be used for prediction.


        

    """

    if isinstance(save_all_labels_in_one_tiff_file, bool):

        if save_all_labels_in_one_tiff_file:

            if tiff_file_name_of_all_labels is None:
                tiff_file_name_of_all_labels = ''  # todo: define the default tiff file name
            elif not isinstance(tiff_file_name_of_all_labels, str):
                raise TypeError('tiff_file_name_of_all_labels must be a string or None')

            # todo save the tiff file
    else:
        raise TypeError('save_all_labels_in_one_tiff_file must be a bool')

    if isinstance(save_different_labels_in_different_tiff_files, bool):
        if save_different_labels_in_different_tiff_files:

            if tiff_file_names_of_different_labels is None:
                tiff_file_names_of_different_labels = []  # todo: define the default tiff file names
            elif not isinstance(tiff_file_name_of_all_labels, str):
                raise TypeError('tiff_file_name_of_all_labels must be a string or None')

            # todo save the tiff files
    else:
        raise TypeError('save_different_labels_in_different_tiff_files must be a bool')


    data = EMData(zarr_path, 'predict', clahe=TRAINING_CONFIG.clahe)
    candidates = None
    
    # Check if there are multiple channels within the raw data.
    # This shouldn't be the case for us as EM data is 'colourblind'.
    if len(data.raw_data.shape) == 3:
        raw_channels = 1
    elif len(data.raw_data.shape) == 4:
        raw_channels = data.raw_data.shape[0]

    # Get an instance of the model
    detection_model = DetectionModel(
                raw_num_channels=raw_channels,
                voxel_size = data.voxel_size
                )

    # Initiate a prediction
    predictor = Prediction(data = data,
                            model = detection_model,
                            input_shape = TRAINING_CONFIG.input_shape, 
                            checkpoint = model_checkpoint)
    
    # Display the border of the output predicition compared to input shape 
    #predictor.print_border_message()
    
    # Get probablities
    ret = predictor.predict_pipeline()
    probs = torch.nn.Softmax(dim=0)(torch.tensor(ret['prediction'].data))
    pos_pred_data = probs[1,:,:,:].detach().numpy()
    neg_pred_data = probs[2,:,:,:].detach().numpy()

    # Post process with hough detector
    hough_detection = HoughDetector(pred_pos = pos_pred_data,
                                    pred_neg = neg_pred_data,
                                    voxel_size = data.voxel_size,
                                    bias = POST_PROCESSING_CONFIG.bias)
    hough_detection.process()
    hough_pred = hough_detection.prediction_result

    hough_pred_pos = np.where(hough_pred == label_pos, label_pos, label_background).astype('int8')
    hough_pred_neg = np.where(hough_pred == label_neg, label_neg, label_background).astype('int8')

    dir_crop = '/home/campus.ncl.ac.uk/ncc222/Projects/neuroscience/data/TIF_data/19-13/subvolume/crops'
    skimage.io.imsave(os.path.join(dir_crop, 'pos.tif'), hough_pred_pos)
    skimage.io.imsave(os.path.join(dir_crop, 'neg.tif'), hough_pred_neg)
    
    candidates = hough_detection.accepted_candidates

    date = datetime.today().strftime('%d_%m_%Y')

    # Create save location
    save_path = create_unique_directory_file(data_path + f'/predict/Predictions/{date}')
    save_location = os.path.relpath(save_path, data_path + '/predict')

    # Save the validation prediction in zarr dictionary. 
    f = zarr.open(data_path + '/predict', mode='r+')
    f[save_location + '/Hough_transformed'] = hough_pred

    for atr in data.raw_data.attrs:
        f[save_location + '/Hough_transformed'].attrs[atr] = data.raw_data.attrs[atr]
    
    return candidates, save_path

if __name__ == "__main__":
        
    data_path = input("Provide path to zarr container: ")

    print("-----")

    model_checkpoint = input("Provide the path to the model checkpoint: ")

    print("-----")

    visualise = input("Would you like to visualise the prediction? (y/n): ")

    while visualise.lower() != 'y' and visualise.lower() != 'n':
        print("-----")
        print("Invalid input. Please enter 'y' or 'n' only.")
        visualise = input("Would you like to visualise the prediction? (y/n): ")

    print("-----")

    candidates, save_location = Apply(zarr_path=data_path, model_checkpoint=model_checkpoint)

    pos_labels = 0 
    neg_labels = 0
    for candidate in candidates:
            if candidate.label == 1:
                pos_labels += 1 
            if candidate.label == 2:
                neg_labels +=1 

    print(f"PC+ predictions: {pos_labels}", f"PC- predictions: {neg_labels}")

    if visualise.lower() == 'y':
            imshow_napari_prediction(data_path, save_location)