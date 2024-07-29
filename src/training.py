import zarr 
import gunpowder as gp  

from src.data_loader import EMData 
from src.model import Model 
from src.gp_filters import AddChannelDim, RemoveChannelDim, TransposeDims 

class Training():
    
    def __init__(self,
                zarr_path: str, 
                clahe=False,  
                input_shape = (44, 96, 96),
                output_shape = (4,56,56)
                ):
        
# QUESTION FOR JAN: how do we determine the input and output shapes?
        
        # Load in the data and create target arrays
        self.zarr_path = zarr_path
        self.training_data = EMData(self.zarr_path, "train", clahe=clahe)
        self.validate_data = EMData(self.zarr_path, "validate", clahe=clahe)
        self.training_data.create_target()
        self.validate_data.create_target()
        self.input_shape = input_shape 
        self.output_shape = output_shape

        # Switch to world units for use with gunpowder
        input_shape = gp.Coordinate(self.input_shape)
        output_shape = gp.Coordinate(self.output_shape)
        self.input_size = self.training_data.voxel_size * input_shape
        self.output_size = self.training_data.voxel_size * output_shape 

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

        self.model = Model(
                    raw_num_channels=self.raw_channels, 
                    input_shape = self.input_shape, 
                    voxel_size = self.training_data.voxel_size
                    ) 

    def training_pipeline( 
                self, 
                augmentations: list,
                batch_size: int, 
                snapshot_every = 0,
                outdir = None
                ):
        
        # A lot of this is based off the pipeline in the again implementation
        
        # Set the model to train mode
        self.model.model.train() 
        
        # Define the gunpowder arrays
        raw = gp.ArrayKey('RAW')
        mask = gp.ArrayKey('MASK')
        target = gp.ArrayKey('TARGET')
        prediction = gp.ArrayKey('PREDICTION')
        
        # Create the source node for pipeline
        if self.training_data.has_mask == True: 
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

# QUESTION FOR JAN: How come I can put infinite pad here? It doesn't like it on target, though. 
# How come again code doesn't pad target?
        pipeline += gp.Pad(raw, None)
        pipeline += gp.Pad(target, (self.input_size-self.output_size)/2 )
        pipeline += gp.Normalize(raw)
        if self.training_data.has_mask == True:
            pipeline += gp.RandomLocation(min_masked=0.1, mask=self.training_data.mask_data)
        else:
            pipeline += gp.RandomLocation()
        
        for augmentation in augmentations:
            pipeline += augmentation

        if self.training_data.has_mask == True:
            loss_inputs = {0: prediction, 1: target, 2: mask}
        else:
            loss_inputs = {0: prediction, 1: target}

        # Note it is important to add channel_dims BEFORE stack, as the CrossEntropyLoss requires this formating 
        if self.channel_dims == 0:
            pipeline += AddChannelDim(raw)

        pipeline += gp.Stack(batch_size)

        pipeline += gp.torch.Train(
            model = self.model.model,
            loss = self.model.loss,
            optimizer = self.model.optimizer,
            inputs = {'input': raw},
            loss_inputs = loss_inputs,
            outputs={0: prediction},
            save_every=100)

        # This needs to be completed later!
        if snapshot_every > 0:
            # get channels first
            pipeline += TransposeDims(raw, (1, 0, 2, 3, 4))

            if self.channel_dims == 0:
                    pipeline += RemoveChannelDim(raw)

        # Create BatchRequest and add arrays with corresponding ROIs
        request = gp.BatchRequest()
        request.add(raw, self.input_size)
        request.add(target, self.output_size)
        request.add(prediction, self.output_size)

        return pipeline, request
    
    def validate_pipeline(self):

        # A lot of this is based off the pipeline in the again implementation

        # Main issue: getting the ROIs correct so that can predict on the full image.
        # Another issue: it appears that batch['target'] is all 0s. Not sure why. 
    
        self.model.model.eval()

        # Define the gunpowder arrays
        raw = gp.ArrayKey('RAW')
        prediction = gp.ArrayKey('PREDICTION')

        # Create scan request (i.e. where each prediction will happen)
        scan_request = gp.BatchRequest()
        scan_request.add(raw, self.input_size)
        scan_request.add(prediction, self.output_size)

        # Create the source node for pipeline
        source = gp.ZarrSource(
                self.validate_data.zarr_path, 
                {
                    raw: self.validate_data.raw_data_path
                },  
                {
                    raw: gp.ArraySpec(interpolatable=True)
                } 
            )

        # Start building the pipeline
        pipeline = source
    
# QUESTION FOR JAN: How come I can put infinite pad here? What is different compared to training?
        pipeline += gp.Pad(raw, None)

        pipeline += gp.Normalize(raw)

        if self.channel_dims == 0:
            pipeline += AddChannelDim(raw)
        
        # This accounts for us having a batch with size 1 
        pipeline += AddChannelDim(raw)

        pipeline += gp.torch.Predict(
                model=self.model.model,
                inputs={'input': raw},
                outputs={0: prediction})
        
        # Remove the created batch dimension
        pipeline += RemoveChannelDim(raw)
        pipeline += RemoveChannelDim(prediction)
        
        if self.channel_dims == 0:
            pipeline += RemoveChannelDim(raw)

        pipeline += gp.Scan(scan_request)

# QUESTION FOR JAN: when I increase the size of this ROI I get the message 
# "Requested TARGET ROI ... lies entirely outside of upstream ROI" 
# See below for more comments on this
        total_request = gp.BatchRequest()
        total_request.add(raw, gp.Coordinate((44,325,600))*self.validate_data.voxel_size)
        total_request.add(prediction, gp.Coordinate((44,325,600))*self.validate_data.voxel_size)


        with gp.build(pipeline):
            batch = pipeline.request_batch(total_request)
            ret = {
                    'raw': batch[raw],
                    'prediction': batch[prediction]
                }
        return ret 


    def load_trained_model(self):
        pass 
