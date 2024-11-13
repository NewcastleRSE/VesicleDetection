import json 
import numpy as np
import os
import pandas as pd
import zarr


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

def save_validations(best_validations: dict, save_path: str, data_path: str):

    """
        Save validations from a training run. 

        Parameters 
        -------------------
        best_valudations (dict):
            Dictionary with keys being score names and values being instances of Validations class.
        save_path (str):
            Path to which to save the run. 
        data_path (str):
            Path to the zarr data that was trained on. 
    """

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

        hough_pred = best_prediction['Hough_transformed']
        save_location = f"{save_path}/{score_name}"
        f_save = zarr.open(save_location + "/prediction", mode='w')
        f_save['Hough_transformed'] = hough_pred

        for atr in f_data['target'].attrs:
            f_save['Hough_transformed'].attrs[atr] = f_data['target'].attrs[atr]