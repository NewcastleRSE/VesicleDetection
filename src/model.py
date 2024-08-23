import torch
from funlib.learn.torch.models import UNet, ConvPass

class DetectionModel(torch.nn.Module):
    def __init__(self,
                raw_num_channels,
                input_shape,
                voxel_size,
                fmaps = 32,
                fmap_inc_factor = 5,
                downsample_factors = [(1,2,2),(1,2,2)],
                padding = 'valid',
                constant_upsample = True
                ):
        
        # Change for an-isotropic data 

        # Look into Segment anything 2 
        
        super(DetectionModel,self).__init__()

        fmaps_in = max(1, raw_num_channels)
        levels = len(downsample_factors) + 1
        dims = len(downsample_factors[0])

        kernel_size_down = [[(3,)*dims, (3,)*dims]]*levels
        kernel_size_up = [[(3,)*dims, (3,)*dims]]*(levels - 1)

        torch.manual_seed(18) 
        
        self.unet = UNet(in_channels = 1,
            num_fmaps = fmaps,
            fmap_inc_factor = fmap_inc_factor,
            kernel_size_down = kernel_size_down,
            kernel_size_up = kernel_size_up,
            downsample_factors = downsample_factors,
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