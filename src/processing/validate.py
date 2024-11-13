import torch
import numpy as np

from src.processing.predict import Prediction
from src.data_loader import EMData
from src.model.model import DetectionModel
from src.processing.post_processing.hough_detector import HoughDetector
from src.processing.post_processing.score_prediction import score_prediction
from src.model.loss import CustomCrossEntropy


class Validations:

    def __init__(self, iteration, scores, predictions, candidates, loss):
        """
            Class for storing validation results.

            Attributes
            -------------------
            iteration:
                The training iteration used for the validation prediction. 
            scores:
                A dictionary containing the scores (recall, precision, fscore)
                for the validation prediction. 
            predictions:
                The prediction data (i.e. the hough transformed prediction). 
            candidates:
                The accepted candidates in the hough transformed prediction.
            loss:
                The loss of the validation run. 
        """
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
    """
        Process for running a validation run. 

        Parameters
        -------------------
        validation_data (EMData):
            A instance of the EMData class which contains the validation data.
        model:
            A instance of the DetectionModel class, which contains the model being 
            trained and validated.
        input_shape:
            The input shape to be used with the model. Should match the input shape
            of the training run. 

        Returns 
        -------------------
        scores (dict):
            A dictionary containing the scores (recall, precision, fscore) of the 
            prediction -- result of the score_prediction function. 
        predictions (dict):
            A dictionary with key "Hough_transformed" who's value is the post processed 
            prediction data.
        candidates (list):
            A list containing the accepted candidates of the post processed predicition. 
            Entries are instances of the HoughCandidate class. 
        val_loss (float):
            The cross entropy loss of the validation run. 
    """

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

    predictions = {'Hough_transformed': hough_pred}
    
    candidates = hough_detection.accepted_candidates
    
    return scores, predictions, candidates, val_loss 
    
    
