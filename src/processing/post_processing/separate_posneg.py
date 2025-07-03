import skimage.io
import sys
import os
# expects the filename to be passed to the script
#
# python -m src.processing.post_processing.separate_posneg filename 
#
# Where filename is the hough transformed output from apply.py

filename = sys.argv[1]
ht = skimage.io.imread(filename)

ht.astype("int32") # reduce default size
pos = ht
neg = ht

neg[neg==1] = 0
pos[pos==2] = 0

input_dir = os.path.dirname(os.path.abspath(filename))
pos_filename = os.path.join(input_dir, "pcplus.tif")
neg_filename = os.path.join(input_dir, "pcneg.tif")

skimage.io.imsave(pos_filename,pos)
skimage.io.imsave(neg_filename,neg)
