import os
import zarr 
import gunpowder as gp
import numpy as np
import yaml
import pandas as pd

from tqdm import tqdm
from datetime import datetime

from src.processing.training import Training, TrainingStatistics
from src.visualisation import imshow_napari_validation
from src.directory_organisor import create_unique_directory_file
from src.processing.post_processing.hough_detector import HoughDetector
from config.load_configs import TRAINING_CONFIG
from src.processing.validate import Validations, validate
from src.save_validations import save_validation_to_JSON, save_validation_to_dataframe
from src.model.loss import CustomCrossEntropy

class Run():

    def __init__(
            self,
            zarr_path: str,
            best_score_name = TRAINING_CONFIG.best_score_name
            ):

        self.zarr_path = zarr_path
        self.training = Training(zarr_path = self.zarr_path)
        self.training_stats = TrainingStatistics()
        self.validations = []
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
                self.training_stats.add_stats(iteration=i, loss=batch.loss, time=train_time)

                # Validate model during training 
                if (i % TRAINING_CONFIG.val_every == 0) and (i>0):
                    print("\n Running validation...")
                    scores, predictions, candidates, val_loss = validate(
                                                            validation_data=self.training.validate_data,
                                                            model = self.training.detection_model,
                                                            input_shape = self.training.input_shape
                                                            )
                    self.validations.append(
                                    Validations(
                                        iteration=i, 
                                        scores=scores, 
                                        predictions=predictions,
                                        candidates=candidates,
                                        loss=val_loss) 
                                    )

                    # Display validation scores to terminal
                    print(scores)

                    # Save best validation statistics
                    if scores[f"{self.best_score_name}_average"] >= self.best_score:
                        self.best_score = scores[f"{self.best_score_name}_average"]
                        self.best_prediction = predictions 
                        self.best_iteration = i
                        self.candidates = candidates

                    print("Resuming training...")
            
            print("Running final validation...")

            train_time = batch.profiling_stats.get_timing_summary('Train', 'process').times[-1]
            self.training_stats.add_stats(iteration=TRAINING_CONFIG.iterations, loss=batch.loss, time=train_time)

            # Compute the final validation after training complete
            scores, predictions, candidates, val_loss = validate(
                                        validation_data=self.training.validate_data,
                                        model = self.training.detection_model,
                                        input_shape = self.training.input_shape
                                        )
            self.validations.append(
                                    Validations(
                                        iteration=TRAINING_CONFIG.iterations, 
                                        scores=scores, 
                                        predictions=predictions,
                                        candidates=candidates, 
                                        loss=val_loss) 
                                    )
            
            # Display validation scores to terminal
            print(scores)

            # Save best validation statistics
            if scores[f"{self.best_score_name}_average"] >= self.best_score:
                self.best_score = scores[f"{self.best_score_name}_average"]
                self.best_prediction = predictions 
                self.best_iteration = TRAINING_CONFIG.iterations
                self.candidates = candidates

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

    # Run training 
    run = Run(data_path)
    run.run_training(checkpoint_path=model_checkpoint_path)
    best_score = run.best_score

    # Check to see if the model has learned enough
    if best_score == 0 or best_score == np.nan:
        print("No prediction obtained. Model needs more training.")

    else:
        best_prediction = run.best_prediction
        best_iteration = run.best_iteration
        candidates = run.candidates
        validations = run.validations

        # Compute number of PC+ and PC- labels
        pos_labels = 0 
        neg_labels = 0
        for candidate in candidates:
            if candidate.label == 1:
                pos_labels += 1 
            if candidate.label == 2:
                neg_labels +=1

        # Create summary dictionary for summary json file
        summary_dict = {}
        summary_dict[f"Best {TRAINING_CONFIG.best_score_name}"] = best_score
        summary_dict["Best iteration"] = best_iteration
        summary_dict["PC+ predictions"] = pos_labels
        summary_dict["PC- predictions"] = neg_labels
        summary_dict["Used pretrained model"] = load_model.lower()
        if load_model.lower() == 'y':
            summary_dict["Model checkpoint used"] = model_checkpoint_path
        
        with open("config/training_config.yaml", "r") as file_object:
            train_config = yaml.full_load(file_object)

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

        # Give the best prediction an attribute indicating which score was used
        f[save_location + '/Positive'].attrs['best_score_name'] = TRAINING_CONFIG.best_score_name
        f[save_location + '/Negative'].attrs['best_score_name'] = TRAINING_CONFIG.best_score_name
        f[save_location + '/Hough_transformed'].attrs['best_score_name'] = TRAINING_CONFIG.best_score_name

        # # Save all validation runs to json file
        # save_validation_to_JSON(validations=validations, 
        #                         json_path=save_path + "/validation_runs", 
        #                         summary_dict=summary_dict, 
        #                         train_dict=train_config)

        save_validation_to_dataframe(validations=validations, training_stats=run.training_stats, csv_path= save_path + "/validation_runs")

        # Get training stats 
        # training_stats_dict = {"Iteration": run.training_stats.iterations, "Loss": run.training_stats.losses, "Time": run.training_stats.times}
        # df = pd.DataFrame(training_stats_dict)
        # print(df)
        
        # Visualise the best prediction in napari
        if visualise.lower() == 'y':
            imshow_napari_validation(data_path, save_location)