import torch
import os 
import zarr

from datetime import datetime

from src.directory_organisor import create_unique_directory_file
from src.processing.predict import Prediction
from src.data_loader import EMData
from src.model.model import DetectionModel
from src.processing.post_processing.hough_detector import HoughDetector
from src.processing.post_processing.score_prediction import score_prediction

from config.load_configs import POST_PROCESSING_CONFIG


class Validations:

    def __init__(self):
        self.iterations = []
        self.scores = []
        self.predictions = []
        self.candidates = []

    def add_validation(self, iteration, scores, predictions, candidates):
        self.iterations.append(iteration)
        self.scores.append(scores)
        self.predictions.append(predictions)
        self.candidates.append(candidates)

def validate(
            validation_data: EMData, 
            model: DetectionModel,
            input_shape
            ):

    predictor = Prediction(
                        data = validation_data, 
                        model = model,
                        input_shape = input_shape
                        )
    
    ret = predictor.predict_pipeline()
    probs = torch.nn.Softmax(dim=0)(torch.tensor(ret['prediction'].data))
    pos_pred_data = probs[1,:,:,:].detach().numpy()
    neg_pred_data = probs[2,:,:,:].detach().numpy()

    hough_detection = HoughDetector(pred_pos = pos_pred_data,
                                    pred_neg = neg_pred_data,
                                    voxel_size = validation_data.voxel_size)
    hough_detection.process()
    hough_pred = hough_detection.prediction_result

    scores = score_prediction(
                            pred = hough_pred,
                            target = validation_data.target_data
                            )
    predictions = {'Positive': pos_pred_data, 
                   'Negative': neg_pred_data, 
                   'Hough_transformed': hough_pred}
    
    candidates = hough_detection.accepted_candidates
    
    return scores, predictions, candidates
    
    
