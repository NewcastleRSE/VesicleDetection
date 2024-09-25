import json 
import numpy as np
import os

from src.processing.validate import Validations


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