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
        
        # Change for an-isotropic data 

        # Look into Segment anything 2 
        
        super(DetectionModel,self).__init__()

        fmaps_in = max(1, raw_num_channels)
        levels = len(downsample_factors) + 1
        dims = len(downsample_factors[0])

        self.downsample_factors = downsample_factors

        self.kernel_size_down = [[(2,) + (3,)*(dims-1), (2,) + (3,)*(dims-1)]]*levels
        self.kernel_size_up = [[(2,) + (3,)*(dims-1), (2,) + (3,)*(dims-1)]]*(levels - 1)

        torch.manual_seed(18) 
        
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

        self.head = ConvPass(fmaps, 3, [(1, 1, 1)], activation=None)

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
    
    border = tuple(((np.array(input_shape) - np.array(conv_out_5))/2).astype(int))


    # print("-----"*5)
    # print(f"The model will divide the image into samples of shape (in voxels) {input_shape} " \
    #         f"and return a corresponding prediction with shape {conv_out_5}.")
    # print(f"The full prediction of the validation image will have a border of shape {border}.")
    # print("-----"*5)
    
    return conv_out_5, border

def ConvOutputShape(input_shape,
                    kernel_size,
                    padding=0, 
                    stride=1
                    ):
    
    in_shape = np.array(input_shape)

    if np.any(in_shape < kernel_size):
        raise ValueError(f'Input shape, {in_shape}, must be greater than kernel size {kernel_size}.')
    
    mid_shape = np.floor((in_shape - kernel_size + 2*padding)/stride +1).astype(int)
    out_shape = np.floor((mid_shape - kernel_size + 2*padding)/stride +1).astype(int)

    return tuple(out_shape)

def DownSampleOutShape(input_shape,
                       downsample_factors):
    in_shape = np.array(input_shape)

    if np.any(in_shape %2 != 0):
        raise ValueError(f'Error when trying to downsample: input shape {in_shape} needs to divide by {downsample_factors}')
    
    out_shape = (in_shape/downsample_factors).astype(int)

    return tuple(out_shape) 

def UpSampleOutShape(input_shape,
                    upsample_factors):
    
    in_shape = np.array(input_shape)
    
    out_shape = (in_shape*upsample_factors).astype(int)

    return tuple(out_shape) 