from src.model.model import DetectionModel
from src.model.model import UnetOutputShape

from config.load_configs import TRAINING_CONFIG

def check_output_shape():
    """
        Check the output shape of the data after being passed through the UNet model.
        Will print the result into terminal window. 
    """

    model = DetectionModel(raw_num_channels=1, 
                           voxel_size=(1,1,1)
                           )
    
    input_shape = TRAINING_CONFIG.input_shape

    output_shape, border = UnetOutputShape(
                                        model = model,
                                        input_shape = input_shape
                                       )
    
    print("-----"*5)
    print(f"The model will divide the image into samples of shape (in voxels) {input_shape} " \
            f"and return a corresponding prediction with shape {output_shape}.")
    print(f"The full prediction of the validation image will have a border of shape {border}.")
    print("-----"*5)

if __name__ == "__main__":
    check_output_shape()