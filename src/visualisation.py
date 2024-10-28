import matplotlib.pyplot as plt
import numpy as np
import napari
import zarr

def imshow_napari_validation(data_path, prediction_path):
    f_data = zarr.open(data_path + "/validate", mode='r')
    raw_data = f_data['raw'][:,:,:]
    target_data = f_data['target'][:,:,:]
    f_prediction = zarr.open(prediction_path, mode='r')
    hough_transformed = f_prediction['Hough_transformed'][:,:,:]

    # Obtain difference between input shape and output shape, to allow alignment in napari
    padding = [int((raw_data.shape[0]-hough_transformed.shape[0])/2), 
               int((raw_data.shape[1]-hough_transformed.shape[1])/2), 
               int((raw_data.shape[2]-hough_transformed.shape[2])/2)]

    viewer = napari.Viewer()
    viewer.add_image(data=raw_data, name='Raw')
    viewer.add_image(data=target_data, name='Target', blending='additive', colormap='inferno')
    viewer.add_image(data=hough_transformed, name='Hough Transformed', blending='additive', colormap='green', translate=padding)
    napari.run()

def imshow_napari_prediction(data_path, prediction_path):
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
