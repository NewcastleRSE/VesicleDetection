import os
import torch
import gunpowder as gp
from datetime import datetime

from src.data_loader import EMData
from src.model.model import DetectionModel, UnetOutputShape
from src.model.loss import CustomCrossEntropy
from src.gp_filters import AddChannelDim, RemoveChannelDim, TransposeDims
from src.directory_organisor import create_unique_directory_file
from config.load_configs import TRAINING_CONFIG

class Training():
    def __init__(self,
                zarr_path: str,
                clahe = TRAINING_CONFIG.clahe,
                training_has_mask = TRAINING_CONFIG.has_mask,
                input_shape = TRAINING_CONFIG.input_shape
                ):
          
        # Load in the data and create target arrays
        self.zarr_path = zarr_path
        self.training_data = EMData(self.zarr_path, "train", clahe=clahe, has_mask = training_has_mask)
        self.validate_data = EMData(self.zarr_path, "validate", clahe=clahe)
        if not self.training_data.has_target:
            self.training_data.create_target()
        if not self.validate_data.has_target:
            self.validate_data.create_target()
        self.input_shape = input_shape

        # Check if there are multiple channels within the raw data.
        # This shouldn't be the case for us as EM data is 'colourblind'.
        if len(self.training_data.raw_data.shape) == 3:
            self.raw_channels = 1
        elif len(self.training_data.raw_data.shape) == 4:
            self.raw_channels = self.training_data.raw_data.shape[0]

        if self.raw_channels == 1:
            self.channel_dims = 0
        else:
            self.channel_dims = 1

        self.detection_model = DetectionModel(
                    raw_num_channels=self.raw_channels,
                    voxel_size = self.training_data.voxel_size
                    )
        
        output_shape, border = UnetOutputShape(
                                        model = self.detection_model,
                                        input_shape = self.input_shape
                                       )

        self.output_shape = output_shape
        self.border = border

        # Obtain shape for prediciton
        if self.input_shape[0] >= self.validate_data.raw_data.shape[0]:
            predict_shape_z = self.input_shape[0]
        else:
            predict_shape_z = self.validate_data.raw_data.shape[0]
        
        if self.input_shape[1] >= self.validate_data.raw_data.shape[1]:
            predict_shape_y = self.input_shape[1]
        else:
            predict_shape_y = self.validate_data.raw_data.shape[1]

        if self.input_shape[2] >= self.validate_data.raw_data.shape[2]:
            predict_shape_x = self.input_shape[2]
        else:
            predict_shape_x = self.validate_data.raw_data.shape[2]

        self.predict_shape = (predict_shape_z, predict_shape_y, predict_shape_x)

        # Switch to world units for use with gunpowder
        input_shape = gp.Coordinate(self.input_shape)
        output_shape = gp.Coordinate(self.output_shape)
        predict_shape = gp.Coordinate(self.predict_shape)
        border_shape = gp.Coordinate(self.border)
        self.input_size = self.training_data.voxel_size * input_shape
        self.output_size = self.training_data.voxel_size * output_shape
        self.predict_size = self.validate_data.voxel_size * predict_shape
        self.border_size = self.validate_data.voxel_size * border_shape
        
        # Weight the cross entropy to account for dominant background labels
        self.loss = CustomCrossEntropy(weight = [0.01, 1.0, 1.0])

        self.optimizer = torch.optim.Adam(self.detection_model.parameters(), lr = 1e-5)

        self.date = datetime.today().strftime('%d_%m_%Y')

    def training_pipeline(
                self,
                augmentations = TRAINING_CONFIG.augmentations,
                batch_size = TRAINING_CONFIG.batch_size,
                snapshot_every = TRAINING_CONFIG.snapshot_every,
                checkpoint_path = TRAINING_CONFIG.checkpoint_path,
                outdir = None
                ):
        
        if checkpoint_path == None:
            # Create directory for saving model checkpoints
            self.checkpoint_path = create_unique_directory_file(f'Model_checkpoints/{self.date}')
            os.makedirs(self.checkpoint_path, exist_ok=True)

        else: 
            self.checkpoint_path = checkpoint_path

        # Set the model to train mode
        self.detection_model.train()

        # Define the gunpowder arrays
        raw = gp.ArrayKey('RAW')
        mask = gp.ArrayKey('MASK')
        target = gp.ArrayKey('TARGET')
        prediction = gp.ArrayKey('PREDICTION')

        # Create the source node for pipeline
        if self.training_data.has_mask:
            source = gp.ZarrSource(
                        self.training_data.zarr_path,
                        {
                            raw: self.training_data.raw_data_path,
                            target: self.training_data.target_data_path,
                            mask: self.training_data.mask_data_path
                        },
                        {
                            raw: gp.ArraySpec(interpolatable=True),
                            target: gp.ArraySpec(interpolatable=False),
                            mask: gp.ArraySpec(interpolatable=False)
                        }
                    )
            
        else:
            source = gp.ZarrSource(
                    self.training_data.zarr_path,
                    {
                        raw: self.training_data.raw_data_path,
                        target: self.training_data.target_data_path
                    },
                    {
                        raw: gp.ArraySpec(interpolatable=True),
                        target: gp.ArraySpec(interpolatable=False)
                    }
                )

        # Start building the pipeline
        pipeline = source

        # Create BatchRequest and add arrays with corresponding ROIs
        request = gp.BatchRequest()
        request.add(raw, self.input_size)
        request.add(target, self.output_size)
        request.add(prediction, self.output_size)
        if self.training_data.has_mask:
            request.add(mask, self.output_size) 

        pipeline += gp.Normalize(raw)
        if self.training_data.has_mask:
            pipeline += gp.RandomLocation(min_masked=0.1, mask=mask)
        else:
            pipeline += gp.RandomLocation()
        
        for augmentation in augmentations:
            pipeline += augmentation

        if self.training_data.has_mask:
            loss_inputs = {0: prediction, 1: target, 2: mask}
        else:
            loss_inputs = {0: prediction, 1: target}

        # Note it is important to add channel_dims BEFORE stack,
        # as the CrossEntropyLoss requires this formating.
        if self.channel_dims == 0:
            pipeline += AddChannelDim(raw)

        pipeline += gp.Stack(batch_size)

        pipeline += gp.torch.Train(
            model = self.detection_model,
            loss = self.loss,
            optimizer = self.optimizer,
            inputs = {'x': raw},
            loss_inputs = loss_inputs,
            outputs={0: prediction},
            checkpoint_basename = self.checkpoint_path + '/model',
            save_every= TRAINING_CONFIG.save_every
            )

        # This needs to be completed later!
        if snapshot_every > 0:
            # get channels first
            pipeline += TransposeDims(raw, (1, 0, 2, 3, 4))

            if self.channel_dims == 0:
                pipeline += RemoveChannelDim(raw)

        return pipeline, request

class TrainingStatistics:

    def __init__(self):
        self.iterations = []
        self.losses = []
        self.times = []

    def add_stats(self, iteration, loss, time):
        self.iterations.append(iteration)
        self.losses.append(loss)
        self.times.append(time) 