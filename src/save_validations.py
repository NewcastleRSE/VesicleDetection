import json 
import numpy as np
import os
import pandas as pd
import csv
import zarr

from src.processing.validate import Validations
from src.processing.training import TrainingStatistics


class NumpyEncoder(json.JSONEncoder):
    """
        Encoder to deal with numpy data types not compatible with json
    """
    def default(self, data):
        if isinstance(data, np.integer):
            return int(data)
        if isinstance(data, np.floating):
            return float(data)
        if isinstance(data, np.ndarray):
            return data.tolist()
        return super(NumpyEncoder, self).default(data)

def save_validation_to_JSON(validations: list[Validations], json_path: str, summary_dict: dict, train_dict: dict):
    for validation in validations:
        # Save scores for each validaton run
        saved_scores = {}
        for k,v in validation.scores.items():
            saved_scores[k] = v

        # Save PC+/PC- logit prediction and Hough_transformed numpy arrays for each validation run
        saved_predictions = {}
        saved_predictions['Positive'] = validation.predictions['Positive']
        saved_predictions['Negative'] = validation.predictions['Negative']
        saved_predictions['Hough_transformed'] = validation.predictions['Hough_transformed']

        # Save candidate locations, maxima scores and labels for each validation run
        saved_candidates = {}
        locations = []
        maxima_scores = []
        labels = []

        for candidate in validation.candidates:
            locations.append(candidate.location)
            maxima_scores.append(candidate.score)
            labels.append(candidate.label)

        saved_candidates['candidate_locations'] = locations
        saved_candidates['candidate_maxima_scores'] = maxima_scores
        saved_candidates['candidate_labels'] = labels

        # Get paths for saving json files
        target_path = json_path 
        iteration_path = json_path + f"/iteration_{validation.iteration}"

        # Create main directory if doesn't exist
        if not os.path.exists(target_path):
            os.makedirs(target_path)

        # Create iteration directory
        if not os.path.exists(iteration_path):
            os.makedirs(iteration_path)
        
        # Save json files
        with open(os.path.join(iteration_path, 'scores.json'), mode='w') as f:
            json.dump(saved_scores, f, cls=NumpyEncoder)

        with open(os.path.join(iteration_path, 'predictions.json'), mode='w') as f:
            json.dump(saved_predictions, f, cls=NumpyEncoder)
        
        with open(os.path.join(iteration_path, 'candidates.json'), mode='w') as f:
            json.dump(saved_candidates, f, cls=NumpyEncoder)

        with open(os.path.join(target_path, 'summary.json'), mode='w') as f:
            json.dump(summary_dict, f, cls=NumpyEncoder)
        
        with open(os.path.join(target_path, 'training_config.json'), mode='w') as f:
            json.dump(train_dict, f, cls=NumpyEncoder)

def save_validation_to_dataframe(validations: list[Validations], training_stats: TrainingStatistics, csv_path: str):
    df_scores_list = []
    df_candidates_list = []
    df_loss_list = []

    for validation in validations:
        # Save scores for each validaton run
        scores_dict = {}
        for k,v in validation.scores.items():
            scores_dict[k] = v
        
        df_scores_iteration = pd.DataFrame(scores_dict, index=[validation.iteration])
        df_scores_list.append(df_scores_iteration)
        

        # # Save PC+/PC- logit prediction and Hough_transformed numpy arrays for each validation run
        # validation_dict['Positive'] = validation.predictions['Positive']
        # validation_dict['Negative'] = validation.predictions['Negative']
        # validation_dict['Hough_transformed'] = validation.predictions['Hough_transformed']

        # Save candidate locations, maxima scores and labels for each validation run
        locations = []
        maxima_scores = []
        labels = []

        for candidate in validation.candidates:
            locations.append(candidate.location)
            maxima_scores.append(candidate.score)
            labels.append(candidate.label)

        candidate_dict = {}
        candidate_dict['location'] = locations
        candidate_dict['score'] = maxima_scores
        candidate_dict['label'] = labels
        candidate_dict['Iteration'] = validation.iteration

        df_candidates_iteration = pd.DataFrame(candidate_dict)
        df_candidates_iteration.set_index('Iteration', inplace=True)
        df_candidates_list.append(df_candidates_iteration)

        validation_loss_dict = {}
        validation_loss_dict['Validation Loss'] = validation.loss
        df_validation_loss = pd.DataFrame(validation_loss_dict, index=[validation.iteration])
        df_loss_list.append(df_validation_loss)
    

    df_scores = pd.concat(df_scores_list)
    df_scores.index.name = 'Iteration'

    df_candidates = pd.concat(df_candidates_list)

    training_stats_dict = {"Iteration": training_stats.iterations, "Training Loss": training_stats.losses}
    df_training_loss = pd.DataFrame(training_stats_dict)
    df_training_loss.set_index('Iteration', inplace=True) 

    df_validation_losses = pd.concat(df_loss_list)
    df_validation_losses.index.name = 'Iteration'

    df_losses = df_training_loss.join(df_validation_losses)

    # Create main directory if doesn't exist
    if not os.path.exists(csv_path):
        os.makedirs(csv_path)

    df_scores.to_csv(csv_path + "/scores.csv", index=True)
    df_candidates.to_csv(csv_path + "/candidates.csv", index=True)
    df_losses.to_csv(csv_path + "/losses.csv", index=True)

    # return df_scores, df_candidates

def save_best_validations(best_validations: dict, save_location):
    # Create iteration directory
    if not os.path.exists(save_location):
        os.makedirs(save_location)

    for k,v in best_validations.items():

        csv_save_location = f"{save_location}/{k}"
        if not os.path.exists(csv_save_location):
            os.makedirs(csv_save_location)

        stats_dict = {"iteration": v.iteration, "scores": v.scores, "loss": v.loss}
        df_stats = pd.DataFrame(stats_dict)

        # Save PC+/PC- logit prediction and Hough_transformed numpy arrays for each validation run
        predictions_dict = {}
        predictions_dict['Positive'] = v.predictions['Positive']
        predictions_dict['Negative'] = v.predictions['Negative']
        predictions_dict['Hough_transformed'] = v.predictions['Hough_transformed']


        # Save candidate locations, maxima scores and labels for each validation run
        locations = []
        maxima_scores = []
        labels = []

        for candidate in v.candidates:
            locations.append(candidate.location)
            maxima_scores.append(candidate.score)
            labels.append(candidate.label)

        candidate_dict = {}
        candidate_dict['location'] = locations
        candidate_dict['score'] = maxima_scores
        candidate_dict['label'] = labels

        df_candidates = pd.DataFrame(candidate_dict)

        # with open(os.path.join(save_location, f'{k}.json'), mode='w') as f:
        #     json.dump(d, f, cls=NumpyEncoder)

        df_stats.to_csv(csv_save_location + "/stats.csv", index=True)
        df_candidates.to_csv(csv_save_location + "/candidates.csv", index=True)

        with open(os.path.join(csv_save_location, 'predictions.json'), mode='w') as f:
            json.dump(predictions_dict, f, cls=NumpyEncoder)

def save_validations(best_validations: dict, best_scores: dict, save_path: str, data_path: str):

    # Create save location directory
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # Load the validation data to get attributes later
    f_data = zarr.open(data_path + "/validate", mode='r')

    for score_name, validation in best_validations.items():

        csv_save_location = f"{save_path}/{score_name}"
        if not os.path.exists(csv_save_location):
            os.makedirs(csv_save_location)

        stats_dict = {"iteration": validation.iteration, "scores": validation.scores, "loss": validation.loss}
        df_stats = pd.DataFrame(stats_dict)

        # Save candidate locations, maxima scores and labels for each validation run
        locations = []
        maxima_scores = []
        labels = []

        for candidate in validation.candidates:
            locations.append(candidate.location)
            maxima_scores.append(candidate.score)
            labels.append(candidate.label)

        candidate_dict = {}
        candidate_dict['location'] = locations
        candidate_dict['score'] = maxima_scores
        candidate_dict['label'] = labels

        df_candidates = pd.DataFrame(candidate_dict)

        df_stats.to_csv(csv_save_location + "/stats.csv", index=True)
        df_candidates.to_csv(csv_save_location + "/candidates.csv", index=True)

        best_prediction = validation.predictions
        best_iteration = validation.iteration
        candidates = validation.candidates
        best_score = best_scores[score_name]

        hough_pred = best_prediction['Hough_transformed']
        save_location = f"{save_path}/{score_name}"
        f_save = zarr.open(save_location + "/prediction", mode='w')
        f_save['Hough_transformed'] = hough_pred

        for atr in f_data['target'].attrs:
            f_save['Hough_transformed'].attrs[atr] = f_data['target'].attrs[atr]