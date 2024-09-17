import os
import zarr 
import gunpowder as gp
import numpy as np

from tqdm import tqdm
from datetime import datetime

from src.processing.training import Training, TrainingStatistics
from src.visualisation import imshow_napari_validation
from src.directory_organisor import create_unique_directory_file
from src.processing.post_processing.hough_detector import HoughDetector
from config.load_configs import TRAINING_CONFIG
from src.processing.validate import Validations, validate

class Run():

    def __init__(
            self,
            zarr_path: str,
            best_score_name = TRAINING_CONFIG.best_score_name
            ):

        self.zarr_path = zarr_path
        self.training = Training(zarr_path = self.zarr_path)
        self.training_stats = TrainingStatistics()
        self.validations = Validations()
        self.best_score_name = best_score_name
        self.best_score = 0.0
        self.candidate = None
        
    def run_training(self, checkpoint_path=None): 
        # Get pipeline and request for training
        pipeline, request = self.training.training_pipeline(checkpoint_path=checkpoint_path)

        # run the training pipeline for interations
        print(f"Starting training for {TRAINING_CONFIG.iterations} iterations...")
        with gp.build(pipeline):
            for i in tqdm(range(TRAINING_CONFIG.iterations)):
                batch = pipeline.request_batch(request)
                train_time = batch.profiling_stats.get_timing_summary('Train', 'process').times[-1]
                self.training_stats.add_stats(iteration=i, loss=batch.loss, time= train_time)

                # Validate model during training 
                if (i % TRAINING_CONFIG.val_every == 0) and (i>0):
                    print("\n Running validation...")
                    scores, predictions, candidates = validate(
                                                            validation_data=self.training.validate_data,
                                                            model = self.training.detection_model,
                                                            input_shape = self.training.input_shape
                                                            )
                    self.validations.add_validation(iteration=i, 
                                                    scores=scores, 
                                                    predictions=predictions, 
                                                    candidates=candidates)

                    print(scores)

                    if scores[f"{self.best_score_name}_average"] >= self.best_score:
                        self.best_score = scores[f"{self.best_score_name}_average"]
                        self.best_prediction = predictions 
                        self.best_iteration = i
                        self.candidates = candidates

                    print("Resuming training...")
            
            print("Running final validation...")
            scores, predictions, candidates = validate(
                                        validation_data=self.training.validate_data,
                                        model = self.training.detection_model,
                                        input_shape = self.training.input_shape
                                        )
            self.validations.add_validation(iteration=i, 
                                            scores=scores, 
                                            predictions=predictions,
                                            candidates=candidates) 

            print(scores)

            if scores[f"{self.best_score_name}_average"] >= self.best_score:
                self.best_score = scores[f"{self.best_score_name}_average"]
                self.best_prediction = predictions 
                self.best_iteration = TRAINING_CONFIG.iterations
                self.candidates = candidates

        if self.best_score == 0 or self.best_score == np.nan:
            return self.best_score, 'N/A', 'N/A'
        
        else:
            return self.best_score, self.best_prediction, self.best_iteration, self.candidates

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
    best_score, best_prediction, best_iteration, candidates = run.run_training(checkpoint_path=model_checkpoint_path)

    if best_prediction == 'N/A':
        print("No prediction obtained. Model needs more training.")

    else:
        print(f"Best {TRAINING_CONFIG.best_score_name}: {best_score}")
        print(f"Best iteration: {best_iteration}")
        pos_labels = 0 
        neg_labels = 0
        for candidate in candidates:
            if candidate.label == 1:
                pos_labels += 1 
            if candidate.label == 2:
                neg_labels +=1 
        
        print(f"PC+ predictions: {pos_labels}", f"PC- predictions: {neg_labels}")

        # Load predictions
        pos_pred = best_prediction['Positive']
        neg_pred = best_prediction['Negative']
        hough_pred = best_prediction['Hough_transformed']

        date = datetime.today().strftime('%d_%m_%Y')

        save_path = create_unique_directory_file(data_path + f'/validate/Predictions/{date}')
        save_location = os.path.relpath(save_path, data_path + '/validate')

        # Save the validation prediction in zarr dictionary. 
        f = zarr.open(data_path + "/validate", mode='r+')
        f[save_location + '/Hough_transformed'] = hough_pred
        f[save_location + '/Positive'] = pos_pred
        f[save_location + '/Negative'] = neg_pred 

        # Copy over attributes from target to predictions
        for atr in f['target'].attrs:
            f[save_location + '/Positive'].attrs[atr] = f['target'].attrs[atr]
            f[save_location + '/Negative'].attrs[atr] = f['target'].attrs[atr]
            f[save_location + '/Hough_transformed'].attrs[atr] = f['target'].attrs[atr]

        f[save_location + '/Positive'].attrs['best_score_name'] = TRAINING_CONFIG.best_score_name
        f[save_location + '/Negative'].attrs['best_score_name'] = TRAINING_CONFIG.best_score_name
        f[save_location + '/Hough_transformed'].attrs['best_score_name'] = TRAINING_CONFIG.best_score_name
        

        if visualise.lower() == 'y':
            imshow_napari_validation(data_path, save_location)