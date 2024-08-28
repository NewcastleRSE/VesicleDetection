import math
import numpy as np
import torch 
import zarr 
import gunpowder as gp
import napari

from datetime import datetime

from src.training import Training
from src.visualisation import imshow, imshow_napari

class Run():

    def __init__(
            self,
            zarr_path: str,
            clahe = False,
            training_has_mask = False
            ):
        self.zarr_path = zarr_path
        self.training = Training(self.zarr_path, clahe=clahe, training_has_mask=training_has_mask)
        self.augmentations = [
            gp.SimpleAugment(transpose_only=(1, 2)),
            gp.ElasticAugment((1, 10, 10), (0, 0.1, 0.1), (0, math.pi/2))
        ]

    def run_training(self, batch_size=1, iterations=1):

        # Get pipeline and request for training
        pipeline, request, raw, target, prediction = self.training.training_pipeline(augmentations=self.augmentations,
                                                            batch_size = batch_size)

        # run the training pipeline for interations
        print(f"Starting training for {iterations} iterations...")
        with gp.build(pipeline):
            for i in range(iterations):
                batch = pipeline.request_batch(request)
                if i % 100 == 0 and i>0:
                    print(f"Completed training iteration {i}")
                    print("Loss: ", batch.loss)

        print("Training complete!")

        # Predict on the validation data
        print("Starting validation...")
        ret = self.training.validate_pipeline()

        return batch, ret, raw, target, prediction 


if __name__ == "__main__":

    data_path = input("Provide path to zarr container: ")

    use_clahe = input("Would you like to use clahe data? (y/n): ")

    if use_clahe.lower() == 'y':
        CLAHE = True
    else:
        CLAHE = False

    has_mask = input("Does your training data have a mask? (y/n): ")

    if has_mask.lower() == 'y':
        HAS_MASK = True
    else:
        HAS_MASK = False

    print(f"Loading data from {data_path}...")

    run = Run(data_path, clahe=CLAHE, training_has_mask=HAS_MASK)
    batch, ret, train_raw, train_target, train_prediction = run.run_training(batch_size=1, iterations=1)
    # Check for convergence issue with batch size (Jan's UNet doesn't have batch normalisation)

    probs = torch.nn.Softmax(dim=0)(torch.tensor(ret['prediction'].data))

    back_pred = probs[0,10:34,20:305,20:580].detach().numpy()
    pos_pred = probs[1,10:34,20:305,20:580].detach().numpy()
    neg_pred = probs[2,10:34,20:305,20:580].detach().numpy()

    total_pred = back_pred + pos_pred + neg_pred
    
    print(batch)
    

    print("---------"*10)
    print(back_pred)

    print("---------"*10)
    print(pos_pred)

    print("---------"*10)
    print(neg_pred)

    # print("---------"*10)
    # print(total_pred)

    # print(np.sum(back_pred))
    # print(np.sum(pos_pred))
    # print(np.sum(neg_pred))
    # print(np.sum(total_pred))
    #print(back_pred)

    #non_zero_indices = np.nonzero(back_pred)

    #print(len(non_zero_indices[0]))
    # print(len(set(non_zero_indices[0])))
    # print(set(non_zero_indices[0]))
    # print(len(set(non_zero_indices[1])))
    # print(set(non_zero_indices[1]))
    # print(len(set(non_zero_indices[2])))
    # print(set(non_zero_indices[2]))

    date = datetime.today().strftime('%d_%m_%Y')


    f = zarr.open(data_path + "/validate", mode='r+')
    f[f'Predictions/{date}/Background'] = back_pred
    f[f'Predictions/{date}/Positive'] = pos_pred
    f[f'Predictions/{date}/Negative'] = neg_pred 

    # Copy over attributes from target to predictions
    for atr in f['target'].attrs:
        f[f'Predictions/{date}/Background'].attrs[atr] = f['target'].attrs[atr]
        f[f'Predictions/{date}/Positive'].attrs[atr] = f['target'].attrs[atr]
        f[f'Predictions/{date}/Negative'].attrs[atr] = f['target'].attrs[atr]

    #imshow_napari(ret)