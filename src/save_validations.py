import json 
import numpy as np
import os

from src.processing.validate import Validations

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)

def save_validation_to_JSON(validations: list[Validations], json_path: str, summary_dict: dict, train_dict: dict):
    for validation in validations:
        saved_scores = {}
        for k,v in validation.scores.items():
            saved_scores[k] = v

        saved_predictions = {}
        saved_predictions['Positive'] = validation.predictions['Positive']
        saved_predictions['Negative'] = validation.predictions['Negative']
        saved_predictions['Hough_transformed'] = validation.predictions['Hough_transformed']

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

        target_path = json_path 
        iteration_path = json_path + f"/iteration_{validation.iteration}"

        if not os.path.exists(target_path):
            os.makedirs(target_path)

        if not os.path.exists(iteration_path):
            os.makedirs(iteration_path)
        
        with open(os.path.join(iteration_path, 'scores.json'), mode='w') as f:
            json.dump(saved_scores, f, cls=NpEncoder)

        with open(os.path.join(iteration_path, 'predictions.json'), mode='w') as f:
            json.dump(saved_predictions, f, cls=NpEncoder)
        
        with open(os.path.join(iteration_path, 'candidates.json'), mode='w') as f:
            json.dump(saved_candidates, f, cls=NpEncoder)

        with open(os.path.join(target_path, 'summary.json'), mode='w') as f:
            json.dump(summary_dict, f, cls=NpEncoder)
        
        with open(os.path.join(target_path, 'training_config.json'), mode='w') as f:
            json.dump(train_dict, f, cls=NpEncoder)