import zarr 
import glob 
import skimage.io 
import numpy as np
import os 

from config.load_configs import TIFF_TO_ZARR_TRAIN_CONFIG


def convert_to_zarr_train():

    f = zarr.open(TIFF_TO_ZARR_TRAIN_CONFIG.output_zarr_path, mode='w')

    # Training Data 

    raw_dir_train = TIFF_TO_ZARR_TRAIN_CONFIG.path_to_raw_tiff_train
    pos_dir_train = TIFF_TO_ZARR_TRAIN_CONFIG.path_to_gt_PC_pos_tiff_train
    neg_dir_train = TIFF_TO_ZARR_TRAIN_CONFIG.path_to_gt_PC_neg_tiff_train

    raw_files_train = sorted(glob.glob(os.path.join(raw_dir_train, '*.tif')))
    pos_files_train = sorted(glob.glob(os.path.join(pos_dir_train, '*.tif')))
    neg_files_train = sorted(glob.glob(os.path.join(neg_dir_train, '*.tif')))

    raw_train = np.array([skimage.io.imread(r) for r in raw_files_train])
    pos_train = np.array([skimage.io.imread(s) for s in pos_files_train]).astype(np.uint64)
    neg_train = np.array([skimage.io.imread(s) for s in neg_files_train]).astype(np.uint64)

    if (raw_train.shape[0] == 1) and (pos_train.shape[0] ==1) and (neg_train.shape[0]==1):
        raw_train = raw_train[0,:]
        pos_train = pos_train[0,:]
        neg_train = neg_train[0,:]

    # bg = 0, pos = 1, neg = 2
    gt_train = (pos_train + neg_train).astype(np.uint64)

    # Replace any labels of 3 with 2
    replace_dict = {3:2}
    for k,v in replace_dict.items():
        gt_train[gt_train==k] = v

    f['train/raw'] = raw_train[:,:,0:70]
    f['train/gt_pos'] = pos_train[:,:,0:70]
    f['train/gt_neg'] = neg_train[:,:,0:70]
    f['train/gt'] = gt_train[:,:,0:70]

    for k,v in TIFF_TO_ZARR_TRAIN_CONFIG.attributes.items():
        f['train/raw'].attrs[k] = v
        f['train/gt'].attrs[k] = v
    
    f['train/gt'].attrs['num_classes'] = 3
    f['train/gt'].attrs['background_label'] = 0

    # Validation Data 
    
    raw_dir_validate = TIFF_TO_ZARR_TRAIN_CONFIG.path_to_raw_tiff_validate
    pos_dir_validate = TIFF_TO_ZARR_TRAIN_CONFIG.path_to_gt_PC_pos_tiff_validate
    neg_dir_validate = TIFF_TO_ZARR_TRAIN_CONFIG.path_to_gt_PC_neg_tiff_validate

    raw_files_validate = sorted(glob.glob(os.path.join(raw_dir_validate, '*.tif')))
    pos_files_validate = sorted(glob.glob(os.path.join(pos_dir_validate, '*.tif')))
    neg_files_validate = sorted(glob.glob(os.path.join(neg_dir_validate, '*.tif')))

    raw_validate = np.array([skimage.io.imread(r) for r in raw_files_validate])
    pos_validate = np.array([skimage.io.imread(s) for s in pos_files_validate]).astype(np.uint64)
    neg_validate = np.array([skimage.io.imread(s) for s in neg_files_validate]).astype(np.uint64)

    if (raw_validate.shape[0] == 1) and (pos_validate.shape[0] ==1) and (neg_validate.shape[0]==1):
        raw_validate = raw_validate[0,:]
        pos_validate = pos_validate[0,:]
        neg_validate = neg_validate[0,:]

    # bg = 0, pos = 1, neg = 2
    gt_validate = (pos_validate + neg_validate).astype(np.uint64)

    # Replace any labels of 3 with 2
    replace_dict = {3:2}
    for k,v in replace_dict.items():
        gt_validate[gt_validate==k] = v

    f['validate/raw'] = raw_validate[:,:,70:]
    f['validate/gt_pos'] = pos_validate[:,:,70:]
    f['validate/gt_neg'] = neg_validate[:,:,70:]
    f['validate/gt'] = gt_validate[:,:,70:]

    for k,v in TIFF_TO_ZARR_TRAIN_CONFIG.attributes.items():
        f['validate/raw'].attrs[k] = v
        f['validate/gt'].attrs[k] = v
    
    f['validate/gt'].attrs['num_classes'] = 3
    f['validate/gt'].attrs['background_label'] = 0

# def convert_to_zarr_predict():

#     f = zarr.open(TIFF_TO_ZARR_CONFIG.output_zarr_path, mode='w')

#     # Training Data 

#     raw_dir = TIFF_TO_ZARR_CONFIG.path_to_raw_tiff

#     raw_files = sorted(glob.glob(os.path.join(raw_dir, '*.tif')))
#     raw = np.array([skimage.io.imread(r) for r in raw_files])

#     if (raw.shape[0] == 1):
#         raw = raw[0,:]

#     f['predict/raw'] = raw[:,:,0:70]
#     for k,v in TIFF_TO_ZARR_CONFIG.attributes.items():
#         f['train/raw'].attrs[k] = v

    
if __name__ == "__main__":
    convert_to_zarr_train()