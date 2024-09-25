import torch 
import os 
import zarr

from datetime import datetime 

from src.data_loader import EMData
from src.processing.predict import Prediction
from src.model.model import DetectionModel
from src.processing.post_processing.hough_detector import HoughDetector
from src.directory_organisor import create_unique_directory_file
from src.visualisation import imshow_napari_prediction

from config.load_configs import TRAINING_CONFIG

def Apply(zarr_path: str, model_checkpoint: str):

    data = EMData(zarr_path, 'predict', clahe=TRAINING_CONFIG.clahe)
    candidates = None
    
    # Check if there are multiple channels within the raw data.
    # This shouldn't be the case for us as EM data is 'colourblind'.
    if len(data.raw_data.shape) == 3:
        raw_channels = 1
    elif len(data.raw_data.shape) == 4:
        raw_channels = data.raw_data.shape[0]

    detection_model = DetectionModel(
                raw_num_channels=raw_channels,
                voxel_size = data.voxel_size
                )

    predictor = Prediction(data = data,
                            model = detection_model,
                            input_shape = TRAINING_CONFIG.input_shape, 
                            checkpoint = model_checkpoint)
    
    predictor.print_border_message()
    
    # Get probablities
    ret = predictor.predict_pipeline()
    probs = torch.nn.Softmax(dim=0)(torch.tensor(ret['prediction'].data))
    pos_pred_data = probs[1,:,:,:].detach().numpy()
    neg_pred_data = probs[2,:,:,:].detach().numpy()

    # Post process with hough detector
    hough_detection = HoughDetector(pred_pos = pos_pred_data,
                                    pred_neg = neg_pred_data,
                                    voxel_size = data.voxel_size)
    hough_detection.process()
    hough_pred = hough_detection.prediction_result
    
    candidates = hough_detection.accepted_candidates

    date = datetime.today().strftime('%d_%m_%Y')

    # Create save location
    save_path = create_unique_directory_file(data_path + f'/predict/Predictions/{date}')
    save_location = os.path.relpath(save_path, data_path + '/predict')

    # Save the validation prediction in zarr dictionary. 
    f = zarr.open(data_path + '/predict', mode='r+')
    f[save_location + '/Hough_transformed'] = hough_pred
    f[save_location + '/Positive'] = pos_pred_data
    f[save_location + '/Negative'] = neg_pred_data

    for atr in data.raw_data.attrs:
        f[save_location + '/Hough_transformed'].attrs[atr] = data.raw_data.attrs[atr]
        f[save_location + '/Positive'].attrs[atr] = data.raw_data.attrs[atr]
        f[save_location + '/Negative'].attrs[atr] = data.raw_data.attrs[atr]
    
    return candidates, save_location

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