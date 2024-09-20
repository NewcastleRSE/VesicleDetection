import zarr 
import glob 
import skimage.io 
import numpy as np
import os 

from config.load_configs import TIFF_TO_ZARR_PREDICT_CONFIG

def convert_to_zarr_predict():

    f = zarr.open(TIFF_TO_ZARR_PREDICT_CONFIG.output_zarr_path, mode='r+')

    # Prediction Data 

    raw_dir = TIFF_TO_ZARR_PREDICT_CONFIG.path_to_raw_tiff
    raw_files = sorted(glob.glob(os.path.join(raw_dir, '*.tif')))
    raw = np.array([skimage.io.imread(r) for r in raw_files])

    if (raw.shape[0] == 1):
        raw = raw[0,:]

    f['predict/raw'] = raw[:]

    for k,v in TIFF_TO_ZARR_PREDICT_CONFIG.attributes.items():
        f['predict/raw'].attrs[k] = v

if __name__ == "__main__":
    convert_to_zarr_predict()