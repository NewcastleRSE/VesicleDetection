import numpy as np
import gunpowder as gp

class AddChannelDim(gp.BatchFilter):

    def __init__(self, array):
        self.array = array

    def process(self, batch, request):

        if self.array not in batch:
            return

        batch[self.array].data = batch[self.array].data[np.newaxis]

class TransposeDims(gp.BatchFilter):

    def __init__(self, array, permutation):
        self.array = array
        self.permutation = permutation

    def process(self, batch, request):

        batch.arrays[self.array].data = \
            batch.arrays[self.array].data.transpose(self.permutation)

class RemoveChannelDim(gp.BatchFilter):

    def __init__(self, array):
        self.array = array

    def process(self, batch, request):

        if self.array not in batch:
            return

        batch[self.array].data = batch[self.array].data[0]
