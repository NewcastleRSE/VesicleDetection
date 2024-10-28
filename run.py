import os
import gunpowder as gp
import numpy as np
import yaml
import torch 
import shutil
import csv

from tqdm import tqdm
from datetime import datetime

from src.processing.training import Training, TrainingStatistics
from src.visualisation import imshow_napari_validation
from src.directory_organisor import create_unique_directory_file
# from src.processing.post_processing.hough_detector import HoughDetector
from config.load_configs import TRAINING_CONFIG
from src.processing.validate import Validations, validate
from src.save_validations import save_validations
# from src.model.loss import CustomCrossEntropy

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

        self.best_scores = {
                            'precision_1': (0,"Iteration "), 
                            'recall_1': (0,"Iteration "), 
                            'fscore_1': (0,"Iteration "),
                            'precision_2': (0,"Iteration "), 
                            'recall_2': (0,"Iteration "), 
                            'fscore_2': (0,"Iteration "),
                            'precision_average': (0,"Iteration "), 
                            'recall_average': (0,"Iteration "), 
                            'fscore_average': (0,"Iteration ")
                            }
        self.best_validations = {}
        
    def run_training(self, model_path=None): 

        date = datetime.today().strftime('%d_%m_%Y')
        self.model_save_path = create_unique_directory_file(f"saved_models/{date}")
        os.makedirs(self.model_save_path + "/model_checkpoints", exist_ok=True)
        os.makedirs(self.model_save_path + "/best_validations", exist_ok=True)
        
        # Load data from previously trained model
        if model_path is not None:
            self.checkpoint_path = f"{model_path}/model_checkpoints/{self.best_score_name}"
            checkpoint = torch.load(self.checkpoint_path, map_location=self.training.device)
            self.training.detection_model.load_state_dict(checkpoint["model_state_dict"])
            self.training.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            with open(model_path + "/best_scores.yaml", "r") as file_object:
                self.best_scores = yaml.full_load(file_object)

            for k,v in self.best_scores.items():
                stripped_v = v[1].replace("*", "")
                self.best_scores[f"{k}"] = (v[0], stripped_v+"*") 

                # Save previous model checkpoints and best validations into new saved model folder
                shutil.copytree(f"{model_path}/model_checkpoints", f"{self.model_save_path}/model_checkpoints", dirs_exist_ok=True)
                shutil.copytree(f"{model_path}/best_validations", f"{self.model_save_path}/best_validations", dirs_exist_ok=True)

            print(self.best_scores)


        # Get pipeline and request for training
        pipeline, request = self.training.training_pipeline() #(checkpoint_path=checkpoint_path)

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
                    
                    # Check for best validation scores
                    for k,v in scores.items(): 
                        if v > self.best_scores[f'{k}'][0]:
                            self.best_scores[f'{k}'] = (v, f"Iteration {i}")
                            self.best_validations[f'{k}'] = Validations(
                                                                    iteration=i, 
                                                                    scores=scores, 
                                                                    predictions=predictions,
                                                                    candidates=candidates,
                                                                    loss=val_loss)
                            
                            torch.save( 
                                {
                                    "model_state_dict": self.training.detection_model.state_dict(),
                                    "optimizer_state_dict": self.training.optimizer.state_dict()
                                }, 
                                f"{self.model_save_path}/model_checkpoints/{k}" 
                            )
                            
                    # Display validation scores to terminal
                    print(self.best_scores)

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

            # Check for best validation scores
            for k,v in scores.items(): 
                if v > self.best_scores[f'{k}'][0]:
                    self.best_scores[f'{k}'] = (v, f"Iteration {TRAINING_CONFIG.iterations}")
                    self.best_validations[f'{k}'] = Validations(
                                                            iteration=TRAINING_CONFIG.iterations, 
                                                            scores=scores, 
                                                            predictions=predictions,
                                                            candidates=candidates,
                                                            loss=val_loss)
                    
                    torch.save( 
                                {
                                    "model_state_dict": self.training.detection_model.state_dict(),
                                    "optimizer_state_dict": self.training.optimizer.state_dict()
                                }, 
                                f"{self.model_save_path}/model_checkpoints/{k}"
                            )
            
            # Display validation scores to terminal
            print(self.best_scores)

    
            self.best_score = self.best_scores[f'{self.best_score_name}']

    def save_run(self, load_model: str):
        
        with open("config/training_config.yaml", "r") as file_object:
            train_config = yaml.full_load(file_object)

        save_validations(best_validations = self.best_validations, 
                         best_scores = self.best_scores, 
                         save_path = f"{self.model_save_path}/best_validations", 
                         data_path = self.zarr_path)
        
        with open(self.model_save_path + "/best_scores.yaml", "w") as file_object:
            yaml.dump(self.best_scores, file_object)
        
        # Compute number of PC+ and PC- labels
        pos_labels = sum(1 for row in csv.DictReader(open(f'{self.model_save_path}/best_validations/{self.best_score_name}/candidates.csv')) if int(row['label']) == 1)
        neg_labels = sum(1 for row in csv.DictReader(open(f'{self.model_save_path}/best_validations/{self.best_score_name}/candidates.csv')) if int(row['label']) == 2)

        # Create summary dictionary for summary json file
        summary_dict = {}
        summary_dict[f"Best {TRAINING_CONFIG.best_score_name}"] = self.best_score
        summary_dict["PC+ predictions"] = pos_labels
        summary_dict["PC- predictions"] = neg_labels
        summary_dict["Used pretrained model"] = load_model.lower()
        if load_model.lower() == 'y':
            summary_dict["Model checkpoint used"] = self.checkpoint_path

        with open(self.model_save_path + "/summary.yaml", "w") as file_object:
            yaml.dump(summary_dict, file_object)

if __name__ == "__main__":

    data_path = input("Provide path to zarr container: ")

    print("-----")
    load_model = input("Would you like to continue training a previous model? (y/n): ")

    while load_model.lower() != 'y' and load_model.lower() != 'n':
        print("-----")
        print("Invalid input. Please enter 'y' or 'n' only.")
        load_model = input("Would you like to continue training a previous model? (y/n): ")

    if load_model.lower() == 'y':
        model_path = input("Provide path to the saved model: ")

        while not os.path.exists(model_path):
            print("-----")
            print("Could not find model checkpoints. Please try again.")
            model_path = input("Provide path to the saved model: ")
    
    else:
        model_path = None

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
    run.run_training(model_path=model_path)

    # Check to see if the model has learned enough
    if run.best_score == 0 or run.best_score == np.nan:
        print("No prediction obtained. Model needs more training.")

    else:
        run.save_run(load_model=load_model)
        
        # Visualise the best prediction in napari
        if visualise.lower() == 'y':
            imshow_napari_validation(data_path=data_path, 
                                     prediction_path=f"{run.model_save_path}/best_validations/{run.best_score_name}/prediction")