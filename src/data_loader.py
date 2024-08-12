import os
import zarr
import gunpowder as gp
import numpy as np 
from skimage import exposure
from torch.utils.data import Dataset


class EMData(Dataset):
    """
        Dataset subclass that loads in the EM data and defines required characteristics. 
    """
    def __init__(self,
                 zarr_path: str,
                 train_validate_predict: str,
                 has_mask = False,
                 clahe = False
                 ):
        
        self.has_mask = has_mask
        
        train_validate_predict = train_validate_predict.lower()
        if train_validate_predict == "train" or train_validate_predict == "validate" or train_validate_predict == "predict":
            
            # Set the data mode type
            self.mode = train_validate_predict
        
            # Locate the zarr file and find the paths to different data types
            self.zarr_path = zarr_path

            # Check if the proviced zarr path is a zarr group
            if '.zgroup' not in os.listdir(self.zarr_path):
                raise FileNotFoundError(f"{self.zarr_path} does not contain required '.zgroup' file.")
            
            # Check if the zarr group has the required train/validate/predict folder
            if self.mode not in os.listdir(self.zarr_path):
                raise FileNotFoundError(f"{self.zarr_path} does not contain a {self.mode} folder.")
            
            # Check if train/validate/predict folder is a zarr group
            if '.zgroup' not in os.listdir(self.zarr_path + "/" + self.mode):
                raise FileNotFoundError(f"{self.zarr_path + "/" + self.mode} does not contain required '.zgroup' file.")

            # Read in the main zarr folder
            self.data = zarr.open(self.zarr_path, mode = 'r')

            # Locate correct raw data and assign
            if clahe:
                if 'raw_clahe' in os.listdir(self.zarr_path + "/" + self.mode):
                    self.raw_data_path = f"/{self.mode}/raw_clahe"
                    self.raw_data = self.data[self.mode]["raw_clahe"]
                else:
                    make_clahe = input(f"'raw_clahe' file not found in {self.zarr_path}/{self.mode}. Would you like to create a clahe file? (y/n) ")
                    if make_clahe.lower() == 'y':
                        self.create_clahe()
                        self.raw_data_path = f"/{self.mode}/raw_clahe"
                        self.raw_data = self.data[self.mode]["raw_clahe"]
                    else:
                        self.raw_data_path = f"/{self.mode}/raw"
                        self.raw_data = self.data[self.mode]["raw"]
            else:
                self.raw_data_path = f"/{self.mode}/raw"
                self.raw_data = self.data[self.mode]["raw"]

            # If train or validate, assign ground-truth data
            if self.mode == "train" or self.mode == "validate":

                if 'gt' not in os.listdir(self.zarr_path + "/" + self.mode):
                    raise FileNotFoundError(f"Ground-truth file is missing in {self.zarr_path}/{self.mode}")
                
                self.gt_data_path = f"/{self.mode}/gt"
                self.gt_data = self.data[self.mode]["gt"]

                # Check if target folder exists and assign target_data_path and target_data
                if 'target' in os.listdir(self.zarr_path + "/" + self.mode):
                    self.target_data_path = f"/{self.mode}/target"
                    self.target_data = self.data[self.mode]["target"]

                # Check if the data has a mask
                if self.has_mask:
                    if 'mask' in os.listdir(self.zarr_path + "/" + self.mode):
                        self.mask_data_path = f"{self.zarr_path}/{self.mode}/mask"
                        self.mask_data = self.data[self.mode]["mask"]
                    else:
                        raise FileNotFoundError(f"Mask file is missing in {self.zarr_path}/{self.mode}")

            # Check raw data has zarr attributes
            if ".zattrs" in os.listdir(self.zarr_path + self.raw_data_path):

                # Check raw data has resolution attribute
                if "resolution" in self.raw_data.attrs:
                    self.resolution = self.raw_data.attrs["resolution"]
                else:
                    raise FileNotFoundError(f"{self.mode} raw data requires resolution attribute.")
                
                if "axes" in self.raw_data.attrs:
                    if self.raw_data.attrs["axes"] == ['z','y','x']:
                        self.axes = self.raw_data.attrs["axes"]
                    else:
                        raise ValueError(f"Raw data axes {self.raw_data.attrs["axes"]}. Axes must be ['z','y','x'].")
                else:
                    raise FileNotFoundError(f"{self.mode} raw data requires axes attribute with orientation [z,y,x].")

                # Check whether raw and gt data have same attributes
                if self.mode == "train" or self.mode == "validate":
                    if ".zattrs" in os.listdir(self.zarr_path + self.gt_data_path):
                        for atr in self.raw_data.attrs:
                            if atr in self.gt_data.attrs:
                                if self.raw_data.attrs[atr] != self.gt_data.attrs[atr]:
                                    raise ValueError(f"{atr} of raw and gt data does not match.")
                    else:
                        raise FileNotFoundError(f"{self.mode} ground-truth data has no attributes. Required attributes: resolution.")
            
            else:
                raise FileNotFoundError(f"{self.mode} raw data has no attributes. Required attributes: resolution.")

            self.voxel_size = gp.Coordinate(self.resolution)

        else:
            raise ValueError("Train_Validate_Test must be either 'Train', 'Validate' or 'Predict'.")
        
    def __len__(self):
        """ 
            Returns the length of the raw data shape 
        """
        return len(self.raw_data.shape)

    def __getitem__(self, index):
        """ 
            Returns a dictionary containing the raw and ground truth data at index 
        """

        if self.mode == "predict":
            raw_data = self.raw_data[index]
            return {"raw": raw_data}

        else:
            # Load the raw and ground truth data
            raw_data = self.raw_data[index]
            gt_data = self.gt_data[index]

            # Return a dictionary containing data
            return {"raw": raw_data, "gt": gt_data}
        
    def create_target(self, data_type = 'int64'):
        """ 
            Create new zarr array, 'target', that is a copy of 'gt' but with different dtype. 
            If 'target' already exists, it will be overwritten. 
        """

        # Open the zarr file in read and write mode
        f = zarr.open(self.zarr_path + "/" + self.mode , mode='r+')

        # create a new zarr array which is equivalent to gt but with changed datatype
        f['target'] = f['gt'].astype(data_type)
        
        # Copy over attributes from gt to target
        for atr in f['gt'].attrs:
            f['target'].attrs[atr] = f['gt'].attrs[atr]
        
        self.target_data_path = f"/{self.mode}/target"
        self.target_data = self.data[self.mode]["target"]

    def create_clahe(self):
        """
            Create a clahe version of the raw data.
        """
        f = zarr.open(self.zarr_path + "/" + self.mode , mode='r+')
        raw = f['raw'] 
        raw_clahe = np.array([
                    exposure.equalize_adapthist(raw[z], kernel_size=128)
                    for z in range(raw.shape[0])
                ], dtype=np.float32)
        f['raw_clahe'] = raw_clahe 
        for atr in f['raw'].attrs:
            f['raw_clahe'].attrs[atr] = f['raw'].attrs[atr] 