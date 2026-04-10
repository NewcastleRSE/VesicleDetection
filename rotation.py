

import itertools

from numpy.array_api import arange
from numpy.ma.core import repeat

from src.visualisation import Viewer
from scipy import ndimage
import numpy as np


n_dims = 3
angles = np.arange(-180, 180, 15, dtype='f').tolist()

angle_comb = itertools.product(angles, repeat=n_dims)

img_raw = np.full([256, 256, 256], 1, dtype=np.uint8)

for angles_i in angle_comb:

    img = img_raw.copy()

    for angle_ij in angles_i:

        img = ndimage.rotate(input=img, angle=angle_ij, axes=(1, 0), reshape=False, mode='constant', cval=0)


# viewer = Viewer(raw_image=data.numpy(), name='Raw Image', opacity=1.0)


# viewer.append_labels(labels=labels, name='labels', opacity=0.4)


# viewer.show()







