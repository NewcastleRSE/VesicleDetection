import torch 

class CustomCrossEntropy(torch.nn.CrossEntropyLoss):

    def __init__(self, weight = None):
        
        if weight is not None:
            weight = torch.tensor(weight)
        
        super(CustomCrossEntropy, self).__init__(weight, reduction='none')

    def forward(self, prediction, target, mask=None):

        loss = super(CustomCrossEntropy, self).forward(prediction, target) 
        
        if mask is not None:
            loss[mask==0] = 0 

        return loss.mean()
          
