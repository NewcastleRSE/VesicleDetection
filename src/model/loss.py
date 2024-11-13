import torch 

class CustomCrossEntropy(torch.nn.CrossEntropyLoss):

    def __init__(self, weight = None):
        """
            Customised version of torch cross entropy loss, designed for classifying 
            voxels within images that contain a mask. Any voxel outside the masked region
            will not contribute to the loss.
        """
        
        if weight is not None:
            weight = torch.tensor(weight)
        
        super(CustomCrossEntropy, self).__init__(weight, reduction='none')

    def forward(self, prediction, target, mask=None):

        loss = super(CustomCrossEntropy, self).forward(prediction, target) 
        
        if mask is not None:
            loss[mask==0] = 0 

        return loss.mean()
          
