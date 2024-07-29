import torch 
from funlib.learn.torch.models import UNet, ConvPass


class Model():

    def __init__(self, 
                raw_num_channels,
                input_shape, 
                voxel_size,
                fmaps = 32,
                fmap_inc_factor = 5, 
                downsample_factors = [(2,2,2),(2,2,2)], 
                padding = 'valid', 
                constant_upsample = True
                ):
        
        fmaps_in = max(1, raw_num_channels)
        levels = len(downsample_factors) + 1
        dims = len(downsample_factors[0])

        kernel_size_down = [[(3,)*dims, (3,)*dims]]*levels
        kernel_size_up = [[(3,)*dims, (3,)*dims]]*(levels - 1)


        self.unet = UNet(in_channels = 1,
            num_fmaps = fmaps,
            fmap_inc_factor = fmap_inc_factor,
            kernel_size_down = kernel_size_down,
            kernel_size_up = kernel_size_up,
            downsample_factors = downsample_factors,
            constant_upsample = True,
            padding = padding,
            voxel_size = voxel_size 
        )

        self.model = torch.nn.Sequential(
                self.unet,
                ConvPass(fmaps, 3, [(1, 1, 1)], activation=None),
                torch.nn.Softmax(dim=1)
                )
        
        # Weight used to reduce effect of background to loss function
        self.loss = torch.nn.CrossEntropyLoss(weight= torch.FloatTensor([0.01, 1.0, 1.0]))

        self.optimizer = torch.optim.Adam(self.model.parameters())

    def save_model(self):
        pass 

    def load_model(self):
        pass 
