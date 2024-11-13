import zarr 
import glob 
import skimage.io 
import numpy as np
import os 

from config.load_configs import TIFF_TO_ZARR_PREDICT_CONFIG

def convert_to_zarr_predict():

    # Check is output directory exists, set mode accordingly
    if os.path.exists(TIFF_TO_ZARR_PREDICT_CONFIG.output_zarr_path):
        mode = 'r+'
    else: 
        mode = 'w'

    f = zarr.open(TIFF_TO_ZARR_PREDICT_CONFIG.output_zarr_path, mode=mode)

    # Get raw tiff data
    raw_dir = TIFF_TO_ZARR_PREDICT_CONFIG.path_to_raw_tiff
    raw_files = sorted(glob.glob(os.path.join(raw_dir, '*.tif')))
    raw = np.array([skimage.io.imread(r) for r in raw_files])

    # Check for stacks of images
    if (raw.shape[0] == 1):
        raw = raw[0,:]

    # Convert to zarr and save
    f['predict/raw'] = raw[:]

    # Set attributes
    for k,v in TIFF_TO_ZARR_PREDICT_CONFIG.attributes.items():
        f['predict/raw'].attrs[k] = v

if __name__ == "__main__":
    convert_to_zarr_predict()