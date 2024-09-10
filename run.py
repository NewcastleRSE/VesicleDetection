import os
import math
import numpy as np
import torch 
import zarr 
import gunpowder as gp
import napari
import time
from tqdm import tqdm

from datetime import datetime

from src.processing.training import Training, TrainingStatistics
from src.processing.predict import Prediction
from src.visualisation import imshow_napari_validation
from src.directory_organisor import create_unique_directory_file
from src.processing.post_processing.hough_detector import HoughDetector

class Run():

    def __init__(
            self,
            zarr_path: str,
            clahe = False,
            training_has_mask = False
            ):
        self.zarr_path = zarr_path
        self.training = Training(self.zarr_path, clahe=clahe, training_has_mask=training_has_mask)
        self.training_stats = TrainingStatistics()
        self.augmentations = [
            gp.SimpleAugment(transpose_only=(1, 2)),
            gp.ElasticAugment((1, 10, 10), (0, 0.1, 0.1), (0, math.pi/2))
        ]

    def run_training(self, batch_size=1, iterations=1, checkpoint_path = None,):

        # Get pipeline and request for training
        pipeline, request = self.training.training_pipeline(augmentations=self.augmentations,
                                                            batch_size = batch_size, 
                                                            checkpoint_path = checkpoint_path)

        # run the training pipeline for interations
        print(f"Starting training for {iterations} iterations...")
        with gp.build(pipeline):
            for i in tqdm(range(iterations)):
                time.sleep(0.01)
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
    use_clahe = input("Would you like to use clahe data? (y/n): ")

    while use_clahe.lower() != 'y' and use_clahe.lower() != 'n':
        print("-----")
        print("Invalid input. Please enter 'y' or 'n' only.")
        use_clahe = input("Would you like to use clahe data? (y/n): ")

    if use_clahe.lower() == 'y':
        CLAHE = True
    elif use_clahe.lower() == 'n':
        CLAHE = False

    print("-----")
    has_mask = input("Does your training data have a mask? (y/n): ")

    while has_mask.lower() != 'y' and has_mask.lower() != 'n':
        print("-----")
        print("Invalid input. Please enter 'y' or 'n' only.")
        has_mask = input("Does your training data have a mask? (y/n): ")

    if has_mask.lower() == 'y':
        HAS_MASK = True
    else:
        HAS_MASK = False

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

    run = Run(data_path, clahe=CLAHE, training_has_mask=HAS_MASK)
    batch, ret = run.run_training(batch_size=1, 
                                iterations=20000, 
                                checkpoint_path = model_checkpoint_path)
    # Check for convergence issue with batch size (Jan's UNet doesn't have batch normalisation)

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