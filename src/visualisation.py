import napari
import zarr

def imshow_napari_validation(data_path, prediction_path):
    """
        Function to display a validation result in napari. The vesicle prediction
        will be overlayed on the raw validation image, along with the hand labelled 
        ground truth. 

        As the UNet crops the output image compared to the input image, the prediction
        will be smaller than the raw validation image and hand labelled ground truth. 
        The prediction will be shown in context of the whole image, and so the user should
        expect a border that contains no prediction. 

        Parameters 
        -------------------
        data_path (str):
            Path to the zarr group containing the data. This should be a zarr group a 'validate'
            zarr group inside it. The validate group should contain two zarr arrays: 'raw' and 
            'target'. 
        prediction_path (str):
            Path to the zarr group containing the prediction. The zarr group should contain a 
            zarr array called 'Hough_transformed' inside it, corresponding to the post-processed 
            prediction on the validation data. 
    """

    # Load the data
    f_data = zarr.open(data_path + "/validate", mode='r')
    raw_data = f_data['raw'][:,:,:]
    target_data = f_data['target'][:,:,:]
    f_prediction = zarr.open(prediction_path, mode='r')
    hough_transformed = f_prediction['Hough_transformed'][:,:,:]

    padding = [0,0,0]

    if hough_transformed.shape < raw_data.shape:
    # Obtain difference between input shape and output shape, to allow alignment in napari
        padding = [int((raw_data.shape[0]-hough_transformed.shape[0])/2), 
                int((raw_data.shape[1]-hough_transformed.shape[1])/2), 
                int((raw_data.shape[2]-hough_transformed.shape[2])/2)]
        
    elif hough_transformed.shape > raw_data.shape: 
        hough_transformed = hough_transformed[0:raw_data.shape[0], 0:raw_data.shape[1], 0:raw_data.shape[2]]
        target_data = target_data[0:raw_data.shape[0], 0:raw_data.shape[1], 0:raw_data.shape[2]]

    viewer = napari.Viewer()
    viewer.add_image(data=raw_data, name='Raw')
    viewer.add_image(data=target_data, name='Target', blending='additive', colormap='inferno')
    viewer.add_image(data=hough_transformed, name='Hough Transformed', blending='additive', colormap='green', translate=padding)
    napari.run()

def imshow_napari_prediction(data_path, prediction_path):
    """
        Function to display a vesicle prediction result overlayed on the raw data. 

        As the UNet crops the output image compared to the input image, the prediction
        will be smaller than the raw input image. The prediction will be shown in 
        context of the whole image, and so the user should expect a border that contains
        no prediction.

        Parameters
        -------------------
        data_path (str):
            Path to the zarr group containing the 'predict' zarr group. The predict zarr group
            should itself contain the 'raw' zarr array, corresponding to the raw data. 
        prediction_path (str):
            Path to the zarr group containing the predicition. This zarr group should contain 
            a zarr array called 'Hough_transformed' inside it, corresponding to the post-processed 
            prediction. 
    """

    f = zarr.open(data_path + '/predict', mode='r')
    raw_data = f['raw'][:,:,:]
    f_prediction = zarr.open(prediction_path, mode='r')
    hough_transformed = f_prediction['/Hough_transformed'][:,:,:]

    # Obtain difference between input shape and output shape, to allow alignment in napari
    padding = [int((raw_data.shape[0]-hough_transformed.shape[0])/2), 
               int((raw_data.shape[1]-hough_transformed.shape[1])/2), 
               int((raw_data.shape[2]-hough_transformed.shape[2])/2)]

    viewer = napari.Viewer()
    viewer.add_image(data=raw_data, name='Raw')
    viewer.add_image(data=hough_transformed, name='Hough Transformed', blending='additive', colormap='inferno', translate=padding)
    napari.run()

if __name__ == "__main__":

    data_path = input("Provide the path to data zarr container: ")
    prediction_path = input("Provide the path to prediction zarr container: ")
    validation_or_predict = input("Is this validation or prediction data? (v/p): ")

    while validation_or_predict.lower() != 'v' and validation_or_predict.lower() != 'p':
        print("-----")
        print("Invalid input. Please enter 'v' or 'p' only.")
        validation_or_predict = input("Is this validation or prediction data? (v/p): ")

    if validation_or_predict.lower() == 'v':
        imshow_napari_validation(data_path=data_path, prediction_path= prediction_path) 
    
    elif validation_or_predict.lower() == 'p':
        imshow_napari_prediction(data_path=data_path, prediction_path= prediction_path) 
