import os
import torch 
import zarr 
import gunpowder as gp

from tqdm import tqdm
from datetime import datetime

from src.processing.training import Training, TrainingStatistics
from src.processing.predict import Prediction
from src.visualisation import imshow_napari_validation
from src.directory_organisor import create_unique_directory_file
from src.processing.post_processing.hough_detector import HoughDetector
from config.load_configs import TRAINING_CONFIG

class Run():

    def __init__(
            self,
            zarr_path: str
            ):

        self.zarr_path = zarr_path
        self.training = Training(zarr_path = self.zarr_path)
        self.training_stats = TrainingStatistics()
        
    def run_training(self): 
        # Get pipeline and request for training
        pipeline, request = self.training.training_pipeline()

        # run the training pipeline for interations
        print(f"Starting training for {TRAINING_CONFIG.iterations} iterations...")
        with gp.build(pipeline):
            for i in tqdm(range(TRAINING_CONFIG.iterations)):
                batch = pipeline.request_batch(request)
                train_time = batch.profiling_stats.get_timing_summary('Train', 'process').times[-1]
                self.training_stats.add_stats(iteration=i, loss=batch.loss, time= train_time)

        # Predict on the validation data
        print(f"Training complete! Starting validation...")
        predictor = Prediction(data = self.training.validate_data, 
                               model = self.training.detection_model,
                               input_shape = self.training.input_shape)
        ret = predictor.predict_pipeline()

        return batch, ret

if __name__ == "__main__":

    data_path = input("Provide path to zarr container: ")

    print("-----")
    load_model = input("Would you like to continue training a previous model? (y/n): ")

    while load_model.lower() != 'y' and load_model.lower() != 'n':
        print("-----")
        print("Invalid input. Please enter 'y' or 'n' only.")
        load_model = input("Would you like to continue training a previous model? (y/n): ")

    if load_model.lower() == 'y':
        model_checkpoint_path = input("Provide path to the models' checkpoints: ")

        while not os.path.exists(model_checkpoint_path):
            print("-----")
            print("Could not find model checkpoints. Please try again.")
            model_checkpoint_path = input("Provide path to the models' checkpoints: ")
    
    else:
        model_checkpoint_path = None 


    print("-----")
    visualise = input("Would you like to visualise the prediction? (y/n): ")

    while visualise.lower() != 'y' and visualise.lower() != 'n':
        print("-----")
        print("Invalid input. Please enter 'y' or 'n' only.")
        visualise = input("Would you like to visualise the prediction? (y/n): ")

    print("-----")
    print(f"Loading data from {data_path}...")

    run = Run(data_path)
    batch, ret = run.run_training()

    # Convert logits output from data to probabilities using softmax.
    probs = torch.nn.Softmax(dim=0)(torch.tensor(ret['prediction'].data))

    # Convert prediction probabilities into numpy arrays
    back_pred = probs[0,:,:,:].detach().numpy()
    pos_pred = probs[1,:,:,:].detach().numpy()
    neg_pred = probs[2,:,:,:].detach().numpy()

    date = datetime.today().strftime('%d_%m_%Y')

    save_path = create_unique_directory_file(data_path + f'/validate/Predictions/{date}')
    save_location = os.path.relpath(save_path, data_path + '/validate')

    # Save the validation prediction in zarr dictionary. 
    f = zarr.open(data_path + "/validate", mode='r+')
    f[save_location + '/Background'] = back_pred
    f[save_location + '/Positive'] = pos_pred
    f[save_location + '/Negative'] = neg_pred 

    # # Copy over attributes from target to predictions
    for atr in f['target'].attrs:
        f[save_location + '/Background'].attrs[atr] = f['target'].attrs[atr]
        f[save_location + '/Positive'].attrs[atr] = f['target'].attrs[atr]
        f[save_location + '/Negative'].attrs[atr] = f['target'].attrs[atr]

    # Post-processing
    hough_detection = HoughDetector(pred_pos=f[save_location + '/Positive'], 
                                    pred_neg=f[save_location + '/Negative'], 
                                    combine_pos_neg=True)

    hough_detection.process(maxima_threshold=50.0)

    f[save_location + '/Hough_transformed'] = hough_detection.prediction_result

    for atr in f[save_location + '/Positive'].attrs:
        f[save_location + '/Hough_transformed'].attrs[atr] = f[save_location + '/Positive'].attrs[atr]

    if visualise.lower() == 'y':
        imshow_napari_validation(data_path, save_location)