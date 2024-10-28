import torch
import numpy as np
import os 
import zarr

from datetime import datetime

from src.directory_organisor import create_unique_directory_file
from src.processing.predict import Prediction
from src.data_loader import EMData
from src.model.model import DetectionModel
from src.processing.post_processing.hough_detector import HoughDetector
from src.processing.post_processing.score_prediction import score_prediction
from src.model.loss import CustomCrossEntropy

from config.load_configs import POST_PROCESSING_CONFIG


class Validations:

    def __init__(self, iteration, scores, predictions, candidates, loss):
        self.iteration = iteration
        self.scores = scores
        self.predictions = predictions
        self.candidates = candidates
        self.loss = loss

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

    # Calculate the loss for the validation 
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    border = predictor.border
    loss_function = CustomCrossEntropy(weight=[0.01, 1.0, 1.0]).to(device)
    target_data = validation_data.target_data[border[0]:-1*border[0], border[1]:-1*border[1], border[2]:-1*border[2]]

    # Add axis for minibatch size and send to device 
    prediction_data_tensor = torch.tensor(ret['prediction'].data[np.newaxis]).to(device)
    target_data_tensor = torch.tensor(target_data[np.newaxis]).to(device)

    loss = loss_function(prediction_data_tensor.float(), target_data_tensor.long())
    val_loss = loss.cpu().detach().numpy()

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
    # predictions = {'Positive': pos_pred_data, 
    #                'Negative': neg_pred_data, 
    #                'Hough_transformed': hough_pred}

    predictions = {'Hough_transformed': hough_pred}
    
    candidates = hough_detection.accepted_candidates
    
    return scores, predictions, candidates, val_loss 
    
    
