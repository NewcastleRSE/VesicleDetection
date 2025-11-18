import napari 
import numpy as np 
import pathlib
from magicgui import magicgui
import os
from enum import Enum
import h5py

if __name__ == "__main__":
    # Load the napari viewer
    viewer = napari.Viewer()

    class Rating(Enum):
        Good = 1
        Bad = 0
        Unsure = -1

    @magicgui(call_button='Load Crop',
              data_path={'label': 'Path to Crop (.h5)', "filter": "*.h5"}
    )
    def load_crop(data_path = pathlib.Path('path/to/crop.h5')) -> napari.types.LayerDataTuple:
        """
            Widget to allow the user to load in a crop that is to be assessed. The data
            must be a TIF file, and the path to this file provided. This can either be entered
            manually, or using the dictionary navigation button.

            Clicking the 'Load' call button will load the provided TIF file into a Napari image 
            layer with the name 'raw'.
        """
        f = h5py.File(data_path, 'r')
        dset_name = data_path.stem  # try dataset name as file name without extension
        if dset_name in f:
            dat = f[dset_name][:]
        else:
            # If not found, just take the first dataset
            first_key = list(f.keys())[0]
            dat = f[first_key][:]
        f.close()
        masked = viewer.add_image(data=dat, name=dset_name, blending='additive', colormap='grey', contrast_limits=[dat.min(), dat.max()])

    @magicgui(call_button='Load Comparison Crop',
              raw_data_path={'label': 'Path to Crop (.h5)', "filter": "*.h5"}
    )
    def load_raw(raw_data_path = pathlib.Path('path/to/raw.h5')) -> napari.types.LayerDataTuple:
        """
            Widget to allow the user to load in a crop that is to be assessed. The data
            must be a TIF file, and the path to this file provided. This can either be entered
            manually, or using the dictionary navigation button.

            Clicking the 'Load' call button will load the provided TIF file into a Napari image 
            layer with the name 'raw'.
        """
        f = h5py.File(raw_data_path, 'r')
        dset_name = raw_data_path.stem  # try dataset name as file name without extension
        if dset_name in f:
            dat = f[dset_name][:]
        else:
            # If not found, just take the first dataset
            first_key = list(f.keys())[0]
            dat = f[first_key][:]
        f.close()
        raw = viewer.add_image(data=dat, name=dset_name, blending='additive', colormap='grey', contrast_limits=[dat.min(), dat.max()])


    @magicgui(call_button='Rate Crop')
    def rate_crop(rating = Rating.Unsure, notes = str("")) -> None:
        """
            Widget to allow the user to rate the currently loaded crop. The rating is an integer
            value that can be set using the spin box. 

            Clicking the 'Rate Crop' call button will print the rating to a csv file alongside the
            masked crop path.
        """
        # Check for existing ratings file, if not present create it
        print(f"Crop rated as: {rating}")
        if not os.path.exists('crop_ratings.csv'):
            with open('crop_ratings.csv', 'w') as f:
                f.write("masked_crop_path,rating,notes\n")
        # Check for any commas in the notes and replace with semicolon to avoid csv issues
        notes = notes.replace(',', ';')
        # Append the new rating to the file
        with open('crop_ratings.csv', 'a') as f:
            layer = viewer.layers.selection.active
            if layer is not None:
                f.write(f"{layer.name},{rating},{notes}\n")
        

    viewer.window.add_dock_widget(load_crop, area='right')
    viewer.window.add_dock_widget(load_raw, area='right')
    viewer.window.add_dock_widget(rate_crop, area='right')


    napari.run()