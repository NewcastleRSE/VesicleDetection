import math
import numpy as np
import zarr 
import gunpowder as gp
import napari

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

    def run_training(self, batch_size= 2, iterations = 1):

        # Get pipeline and request for training
        pipeline, request, raw, target, prediction = self.training.training_pipeline(augmentations=self.augmentations,
                                                            batch_size = batch_size)

        # run the training pipeline for interations
        print(f"Starting training for {iterations} iterations...")
        with gp.build(pipeline):
            for i in range(iterations):
                batch = pipeline.request_batch(request)

        print("Training complete!")

        # Predict on the validation data
        print("Starting validation...")
        ret = self.training.validate_pipeline()

        return batch, ret, raw, target, prediction 


if __name__ == "__main__":

    data_path = input("Provide path to zarr container: ")

    print(f"Loading data from {data_path}...")

    use_clahe = input("Would you like to use clahe data? (y/n): ")

    if use_clahe.lower() == 'y':
        CLAHE = True
    else:
        CLAHE = False

    run = Run(data_path, clahe=CLAHE)
    batch, ret, train_raw, train_target, train_prediction = run.run_training(batch_size=1, iterations=1)
    # Check for convergence issue with batch size (Jan's UNet doesn't have batch normalisation)

    # Output the predicitions for background, PC+ and PC-
    # Total should be 1 everywhere. Something seems to be wrong!

    back_pred = ret['prediction'].data[0,:,:,:]
    pos_pred = ret['prediction'].data[1,:,:,:]
    neg_pred = ret['prediction'].data[2,:,:,:]
    total_pred = back_pred + pos_pred + neg_pred
    
    pred_shape = ret['prediction'].data.shape

    print(batch)
    #print(ret) 
    print(pred_shape)

    print(np.sum(back_pred))
    print(np.sum(pos_pred))
    print(np.sum(neg_pred))
    print(np.sum(total_pred))
    #print(back_pred)

    non_zero_indices = np.nonzero(back_pred)

    #print(len(non_zero_indices[0]))
    print(len(set(non_zero_indices[0])))
    print(set(non_zero_indices[0]))
    print(len(set(non_zero_indices[1])))
    print(set(non_zero_indices[1]))
    print(len(set(non_zero_indices[2])))
    print(set(non_zero_indices[2]))

    #print(ret['prediction'].data[0,10:34,20:305,20:580])
    

    back_train = batch[train_prediction].data[0,0,:,:,:]
    pos_train = batch[train_prediction].data[0,1,:,:,:]
    neg_train = batch[train_prediction].data[0,2,:,:,:]

    # f = zarr.open(data_path + "/validate", mode='r+')
    # f['Background'] = back_pred
    # f['Positive'] = pos_pred
    # f['Negative'] = neg_pred 

    total_train = back_train + pos_train + neg_train

    # print('--------------------------'*5)
    # print(back_train) 
    # print('--------------------------'*5)
    # print(pos_train) 
    # print('--------------------------'*5)
    # print(neg_train) 
    # print('--------------------------'*5)
    # print(total_train)

    # if not 1 in back_train:
    #     print('its not all background!')
    # else:
    #     print('All background, sad!')

    # if not 0 in pos_train: 
    #     print("pos train success") 
    # else:
    #     print("pos train all 0")

    # if not 0 in neg_train: 
    #     print("neg train success") 
    # else:
    #     print("neg train all 0")

    #imshow_napari(ret)