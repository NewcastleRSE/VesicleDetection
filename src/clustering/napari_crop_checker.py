import napari 
import numpy as np 
import pathlib
from magicgui import magicgui, widgets
import os
from enum import Enum
import h5py
import glob
import pandas as pd

if __name__ == "__main__":
    # Load the napari viewer
    viewer = napari.Viewer()

    #---- Helper Functions ----#

    class Rating(Enum):
        Good = 1
        Bad = 0
        Unsure = -1

    class FolderState:
        def __init__(self):
            self.root_path = None
            self.crop_ids = []
            self.current_idx = 0
            self.csv_path = None
            self.df = None # This will hold our ratings table

    state = FolderState()

    def setup_four_panel_view(viewer, layer):
        """Create a 4-panel orthogonal viewer layout for the given image layer."""

        # Ensure data is 3D
        if layer.data.ndim != 3:
            print("4-panel orthogonal view requires a 3D image.")
            return

        # === Create axis-swapped images ===
        xy_view = layer

        xz_view = viewer.add_image(
            layer.data.transpose(1, 0, 2),
            name=layer.name + "_XZ",
            contrast_limits=layer.contrast_limits,
            colormap=layer.colormap,
            blending='additive'
        )

        yz_view = viewer.add_image(
            layer.data.transpose(2, 0, 1),
            name=layer.name + "_YZ",
            contrast_limits=layer.contrast_limits,
            colormap=layer.colormap,
            blending='additive'
        )

        xy2_view = viewer.add_image(
            layer.data,
            name=layer.name + "_XY2",
            contrast_limits=layer.contrast_limits,
            colormap=layer.colormap,
            blending='additive',
            rendering='attenuated_mip',
            depiction='volume'
        )

        # === Put viewer into 4-panel grid ===
        viewer.grid.enabled = True
        viewer.grid.shape = (2, 2)

        # Force napari to rearrange views
        viewer.reset_view()

        # === IMPORTANT: Remove layer.dims references ===
        # Orthogonal views are created by supplying transposed data
        # NOT by altering dims.order (which only exists on viewer)

        print("4-panel orthogonal view ready.")


    def load_pair(masked_path, raw_path):
        """Helper to clear viewer and load the specific h5 pair."""
        viewer.layers.clear()
        
        # Load Masked
        with h5py.File(masked_path, 'r') as f:
            dset = list(f.keys())[0]
            masked_data = f[dset][:]
        
        # Load Raw
        with h5py.File(raw_path, 'r') as f:
            dset = list(f.keys())[0]
            raw_data = f[dset][:]

        # Add to viewer
        # m_layer = viewer.add_image(masked_data, name=f"Masked_{masked_path.stem}", colormap='green', blending='additive')
        # r_layer = viewer.add_image(raw_data, name=f"Raw_{raw_path.stem}", colormap='grey', blending='additive')
        
        # # Add center marker
        # center = np.array(masked_data.shape) / 2
        # viewer.add_points([center], name="center marker", size=10, face_color="red")
        
        # # Trigger your 4-panel view
        # setup_four_panel_view(viewer, m_layer)
        # setup_four_panel_view(viewer, r_layer)
        load_crop(masked_path)
        load_raw(raw_path)

    #---- MagicGUI Widgets ----#

    @magicgui(
            call_button="Initialize Folder",
            folder_path={"label": "Select Folder", "mode": "d"},
        )
    def folder_navigator(folder_path=pathlib.Path.cwd()):
        """Finds all clusters and prepares the list."""
        state.root_path = folder_path
        masked_dir = folder_path / "masked"
        
        # Find all files matching the pattern and extract IDs
        files = glob.glob(str(masked_dir / "cluster_*_masked.h5"))
        # Sort numerically by extracting the number from the filename
        state.crop_ids = sorted([os.path.basename(f).split('_')[1] for f in files], key=int)
        state.current_idx = 0

        folder_name = os.path.basename(folder_path)
        state.csv_path = folder_path / f"{folder_name}_ratings.csv"

        if os.path.exists(state.csv_path):
            state.df = pd.read_csv(state.csv_path)
            # Convert ID column to string to ensure matching works
            state.df['id'] = state.df['id'].astype(str)
            data = {
                "id": state.df['id'],
                "rating": state.df['rating'],
                "notes": state.df['notes'],
                "masked_path": state.df['masked_path']
            }
        else:
            # Create a fresh table with all IDs pre-populated
            data = {
                "id": state.crop_ids,
                "rating": ["Unsure"] * len(state.crop_ids),
                "notes": [""] * len(state.crop_ids),
                "masked_path": [f"masked/cluster_{cid}_masked.h5" for cid in state.crop_ids]
            }
        state.df = pd.DataFrame(data)
        state.df.to_csv(state.csv_path, index=False)
        
        if state.crop_ids:
            show_current_crop()
        else:
            print("No matching crops found in folder!")


    def show_current_crop():
            cid = state.crop_ids[state.current_idx]
            m_path = state.root_path / "masked" / f"cluster_{cid}_masked.h5"
            r_path = state.root_path / "raw" / f"cluster_{cid}_raw.h5"
            
            if m_path.exists() and r_path.exists():
                load_pair(m_path, r_path)
                # Update the label on our custom button container
                status_label.value = f"Crop {state.current_idx + 1} of {len(state.crop_ids)} (ID: {cid})"
            else:
                print(f"Missing one of the pair for ID {cid}")


    # --- Create Next/Prev Buttons ---
    @magicgui(call_button="Next Crop >>")
    def next_crop():
        if state.current_idx < len(state.crop_ids) - 1:
            state.current_idx += 1
            show_current_crop()

    @magicgui(call_button="<< Prev Crop")
    def prev_crop():
        if state.current_idx > 0:
            state.current_idx -= 1
            show_current_crop()

    # Create a container to hold the navigation status
    status_label = widgets.Label(value="No folder loaded")

    next_prev_widget = widgets.Container(
        widgets=[
            prev_crop, 
            next_crop, 
            status_label # Use the variable here
        ],
        layout="horizontal",
        labels=False
    )

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
        center = np.array(dat.shape) / 2
        points = np.array([center])
        viewer.add_points(
            points,
            name="center marker",
            size=10,
            face_color="red"
            ) 
        masked = viewer.add_image(data=dat, name=dset_name, blending='additive', colormap='grey', contrast_limits=[dat.min(), dat.max()])
        setup_four_panel_view(viewer, masked)

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
        setup_four_panel_view(viewer, raw)

    @magicgui(call_button='Save Rating',
              notes={"label": "Notes:"})
    def rate_crop(rating = Rating.Unsure, notes = str("")) -> None:
        """
            Widget to allow the user to rate the currently loaded crop. The rating is an integer
            value that can be set using the spin box. 

            Clicking the 'Rate Crop' call button will print the rating to a csv file alongside the
            masked crop path.
        """
        if state.df is None:
            print("No folder loaded, cannot save rating!")
            return
        
        current_id = str(state.crop_ids[state.current_idx])
    
        # Update the local DataFrame (find row where 'id' matches)
        state.df.loc[state.df['id'] == current_id, 'rating'] = rating.name
        state.df.loc[state.df['id'] == current_id, 'notes'] = notes.replace(',', ';')
        
        # Save the whole thing back to CSV
        state.df.to_csv(state.csv_path, index=False)
        
        print(f"Updated ID {current_id} with {rating.name}")
        
        # Optional: Automatically move to next after rating
        next_crop()


    
    #viewer.window.add_dock_widget(load_crop, area='right')    
    #viewer.window.add_dock_widget(load_raw, area='right')

    viewer.window.add_dock_widget(folder_navigator, area='right', name="1. Setup")
    viewer.window.add_dock_widget(next_prev_widget, area='right', name="2. Navigation")
    viewer.window.add_dock_widget(load_crop, area='right', name="Optional: Load Individual Crop")
    viewer.window.add_dock_widget(rate_crop, area='right', name="3. Scoring")

    napari.run()