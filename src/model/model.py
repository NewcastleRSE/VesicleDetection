import torch
import numpy as np
import math
from funlib.learn.torch.models import UNet, ConvPass

from config.load_configs import MODEL_CONFIG

class DetectionModel(torch.nn.Module):
    def __init__(self,
                raw_num_channels,
                voxel_size,
                fmaps = MODEL_CONFIG.fmaps,
                fmap_inc_factor = MODEL_CONFIG.fmap_inc_factors,
                downsample_factors = MODEL_CONFIG.downsample_factors,
                padding = MODEL_CONFIG.padding,
                constant_upsample = MODEL_CONFIG.constant_upsample
                ):
        
        """
            Machine learning model for vesicle detection pipeline. Consists of a 
            UNet followed by a ConvPass with 3 outputs. Per voxel it returns a 3D 
            array who's entries are the probabilities that voxel is background, PC+
            or PC-, respectively. 

            Parameters
            -------------------
            raw_num_channels (int):
                The number of channels in the input raw data. 
            voxel_size (tuple(int)):
                The size of a voxel in the input data, in physical units.
            fmaps (int):
                The number of feature maps in the first layer of the UNet. 
                Default set using model_config.yaml file.
            fmap_inc_factor (int):
                The multiplicative factor for feature maps between layers.
                If layer n has k feature maps, layer (n+1) will have 
                (k * fmap_inc_factor) feature maps. Default set using 
                model_config.yaml file.
            downsample_factors (list(tuple)):
                A list of two tuples. The first sets the downsampling factor 
                in (z,y,x) between layers of the UNet. The second sets the 
                upsampling factor in (z,y,x). Default set using model_config.yaml 
                file.
            padding (str):
                How to pad convolutions within the UNet. Options: 'same' or 'valid'. 
                Default set using model_config.yaml file.
            constant_upsample (bool):
                Controls upsampling layers in the UNet. If true, will perform a constant 
                upsampling instead of a transposed convolution. Default set using 
                model_config.yaml file.
        """
        
        super(DetectionModel,self).__init__()

        fmaps_in = max(1, raw_num_channels)
        levels = len(downsample_factors) + 1
        dims = len(downsample_factors[0])

        self.downsample_factors = downsample_factors

        self.kernel_size_down = [[(3,) + (3,)*(dims-1), (3,) + (3,)*(dims-1)]]*levels
        self.kernel_size_up = [[(3,) + (3,)*(dims-1), (3,) + (3,)*(dims-1)]]*(levels - 1)

        torch.manual_seed(18) 
        
        # Define the UNet part of the model
        self.unet = UNet(in_channels = 1,
            num_fmaps = fmaps,
            fmap_inc_factor = fmap_inc_factor,
            kernel_size_down = self.kernel_size_down,
            kernel_size_up = self.kernel_size_up,
            downsample_factors = self.downsample_factors,
            constant_upsample = constant_upsample,
            padding = padding,
            voxel_size = voxel_size
        )

        # Define the convolution head of the model
        self.head = ConvPass(fmaps, 3, [(1, 1, 1)], activation=None)

        # Define the complete model 
        self.total_model = torch.nn.Sequential(
                    self.unet,
                    self.head
                    )

    def forward(self, x):
        x = self.total_model(x)
        return x
    
def UnetOutputShape(
                    model: DetectionModel,
                    input_shape, 
                    padding = 0,
                    stride = 1
                    ):
    """
        Obtain the output shape of an image passed through the vesicle detection 
        model -- the UNet performs downsampling and upsampling, so the resulting output
        image can be smaller than the input image. The UNet is assumed to contain 2 
        downsampling layers and two upsampling layers, with convolution layers between 
        every other layer for a total of 9 layers (C = conv, D = down, U = up):
        C -> D -> C -> D -> C -> U -> C -> U -> C

        Parameters 
        -------------------
        model (DetectionModel Class):
            The model to pass the image through. 
        input_shape (tuple):
            The shape (in voxels) of the input image. 
        padding (int):
            The padding used in convolutions of the UNet. 
        string (int):
            The stride used in the convolutions of the UNet.

        Returns
        -------------------
        conv_5_out: 
            The shape of the output shape after the last convolution layer. 
        border: 
            The size of border of the output shape compared to the input shape.
    """

    # Get model specific factors
    kernel_size_down = tuple(model.kernel_size_down[0][0])
    kernel_size_up = tuple(model.kernel_size_up[0][0])
    downsample_factors = model.downsample_factors[0]
    upsample_factors = model.downsample_factors[1]

    conv_out_1 = ConvOutputShape(input_shape=input_shape,
                                 kernel_size=kernel_size_down,
                                 padding=padding, 
                                 stride=stride
                                 )
    
    down_out_1 = DownSampleOutShape(input_shape=conv_out_1,
                                    downsample_factors=downsample_factors)
    
    conv_out_2 = ConvOutputShape(input_shape=down_out_1,
                                 kernel_size=kernel_size_down,
                                 padding=padding, 
                                 stride=stride
                                 )
    
    down_out_2 = DownSampleOutShape(input_shape=conv_out_2,
                                    downsample_factors=downsample_factors)
    
    conv_out_3 = ConvOutputShape(input_shape=down_out_2,
                                 kernel_size=kernel_size_down,
                                 padding=padding, 
                                 stride=stride
                                 )
    
    up_out_1 = UpSampleOutShape(input_shape=conv_out_3, 
                                upsample_factors=upsample_factors)
    
    conv_out_4 = ConvOutputShape(input_shape=up_out_1,
                                 kernel_size=kernel_size_up,
                                 padding=padding, 
                                 stride=stride
                                 )
    
    up_out_2 = UpSampleOutShape(input_shape=conv_out_4, 
                                upsample_factors=upsample_factors)
    
    conv_out_5 = ConvOutputShape(input_shape=up_out_2,
                                 kernel_size=kernel_size_up,
                                 padding=padding, 
                                 stride=stride
                                 )
    
    # Compute the border of the output shape compared to the input shape.
    border = tuple(((np.array(input_shape) - np.array(conv_out_5))/2).astype(int))
    
    return conv_out_5, border

def ConvOutputShape(input_shape,
                    kernel_size,
                    padding=0, 
                    stride=1
                    ):
    """
        Computes the shape of an image after a convolution layer.
    """
    
    in_shape = np.array(input_shape)

    if np.any(in_shape < kernel_size):
        raise ValueError(f'Input shape, {in_shape}, must be greater than kernel size {kernel_size}.')
    
    mid_shape = np.floor((in_shape - kernel_size + 2*padding)/stride +1).astype(int)
    out_shape = np.floor((mid_shape - kernel_size + 2*padding)/stride +1).astype(int)

    return tuple(out_shape)

def DownSampleOutShape(input_shape,
                       downsample_factors):
    """
        Computes the shape of an image after a downsampling layer.
    """

    in_shape = np.array(input_shape)

    if np.any(in_shape %2 != 0):
        raise ValueError(f'Error when trying to downsample: input shape {in_shape} needs to divide by {downsample_factors}')
    
    out_shape = (in_shape/downsample_factors).astype(int)

    return tuple(out_shape) 

def UpSampleOutShape(input_shape,
                    upsample_factors):
    """
        Computes the shape of an image after a upsampling layer.
    """
    
    in_shape = np.array(input_shape)
    
    out_shape = (in_shape*upsample_factors).astype(int)

    return tuple(out_shape) 