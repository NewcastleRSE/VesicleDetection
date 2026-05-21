import napari 
import numpy as np 
import pathlib
from magicgui import magicgui, widgets
import os
from enum import Enum
import h5py
import glob
import pandas as pd
from qtpy.QtWidgets import QListWidget
from qtpy.QtCore import Qt

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
            self.cluster_locs = None  # Global coordinates from npz
            self.cluster_labels = None # IDs from npz
            self.show_labels = True

    state = FolderState()


    def load_4panel_view(path_or_points, type="masked"):
        """
                Helper to clear viewer and load the specific h5 pair in a 4-panel layout. The type argument is used to determine whether we are loading the masked or raw version of the crop, which will be used to construct the file paths for both the masked and raw data. The function will then load both datasets, extract nearby cluster points, and display everything in a 4-panel orthogonal layout with appropriate labels.

        Args:
            path_or_points (str or np.ndarray): Path to the h5 file or array of points
            type (str, optional): _description_. Defaults to "masked". Exepectes either "masked", "raw", or "points".
        """

        if type not in ["masked", "raw", "points"]:
            print(f"Invalid type '{type}' specified. Must be 'masked', 'raw', or 'points'.")
            return
        
        if type == "points" and (state.cluster_locs is None or state.cluster_labels is None):
            print("Cluster location data not available. Cannot display points.")
            return


        if type=="points":
            pts, ids = path_or_points[0], path_or_points[1]
        else:   
            with h5py.File(path_or_points, 'r') as f:
                data = f[list(f.keys())[0]][:]

        # 3. Explicitly define each panel
        # Format: (Row, Col, Image_Transpose_Tuple, Point_Column_Order, Suffix)
        panels = [
            (0, 0, (0, 1, 2), [0, 1, 2], "XY"), # Standard
            (0, 1, (1, 0, 2), [1, 0, 2], "XZ"), # Swap Z and Y
            (1, 0, (2, 1, 0), [2, 1, 0], "YZ"), # Swap Z and X
            (1, 1, (0, 2, 1), [0, 2, 1], "3D")  # Volume
        ]

        for row, col, img_axes, pt_axes, suffix in panels:

            if type == "points":
                # For points-only view, we just add an empty image layer to hold the points
                img = viewer.add_points(
                    pts[:, pt_axes], 
                    name=f"Labels_{suffix}",
                    features={'cluster_id': ids},
                    text={'string': '{cluster_id}', 'color': 'yellow', 'size': 12},
                    face_color='transparent',
                    border_color='yellow',
                    size=5
                )
                img.visible = state.show_labels
            else:
                # Add Images
                img = viewer.add_image(data.transpose(img_axes), name=f"{type}_{suffix}", colormap='gray', blending='additive', opacity=0.6)
                img.grid_index = (row, col)
            
                if suffix == "3D":
                    img.rendering = 'attenuated_mip'
                    img.depiction = 'volume'

            img.grid_index = (row, col)

        viewer.grid.enabled = True
        viewer.grid.shape = (2, 2)
        viewer.reset_view()

    def get_nearby_clusters(current_cid, shape):
        if state.cluster_locs is None: return None, None
        
        # 2. Find Global Center of Anchor Cluster
        mask = state.cluster_labels == int(current_cid)
        if not np.any(mask): return None, None
        global_center = state.cluster_locs[mask].mean(axis=0)
        
        half_shape = np.array(shape) / 2
        nearby_points = []
        nearby_ids = []

        # 3. Filter Clusters in Crop Range
        for cid in np.unique(state.cluster_labels):
            if cid == -1: continue
            c_center = state.cluster_locs[state.cluster_labels == cid].mean(axis=0)
            
            diff = c_center - global_center
            if np.all(np.abs(diff) <= half_shape):
                # Transform to local crop space
                nearby_points.append(diff + half_shape)
                nearby_ids.append(str(cid))
                
        return (np.array(nearby_points), nearby_ids) if nearby_points else (None, None)


    @magicgui(call_button='Load Crop',
              data_path={'label': 'Path to Crop (.h5)', "filter": "*.h5"}
    )
    def load_crop(data_path = pathlib.Path('path/to/crop.h5'), cluster_labels=None) -> napari.types.LayerDataTuple:
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
        load_4panel_view(data_path, type=dset_name)


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
        load_crop(raw_data_path, cluster_labels=False)

    #---- MagicGUI Widgets ----#

    @magicgui(
            call_button="Initialize Folder",
            folder_path={"label": "Select Folder", "mode": "d"},
            npz_path={"label": "Cluster NPZ", "filter": "*.npz"},
        )
    def folder_navigator(folder_path=pathlib.Path.cwd(), npz_path=pathlib.Path.cwd()):
        """Finds all clusters and prepares the list."""
        state.root_path = folder_path
        masked_dir = folder_path / "masked"

            # --- Load Cluster Info ---
        if os.path.exists(npz_path):
            data = np.load(npz_path)
            state.cluster_locs = data['locs']
            state.cluster_labels = data['labels']
        
        # Find all files matching the pattern and extract IDs
        files = glob.glob(str(masked_dir / "*cluster_*_masked.h5"))
        # Sort numerically by extracting the number from the filename
        # need this to find the crop IDs, which are assumed to be in the format "*cluster_{id}_masked.h5" but there can be any prefix, including underscores, before "cluster"
        state.crop_ids = sorted([os.path.basename(f).split('cluster_')[1].split('_masked.h5')[0] for f in files], key=int)
        # state.crop_ids = sorted([os.path.basename(f).split('_')[1] for f in files], key=int)
        state.current_idx = 0

        folder_name = os.path.basename(folder_path)
        state.csv_path = folder_path / f"{folder_name}_ratings.csv"

        if os.path.exists(state.csv_path):
            state.df = pd.read_csv(state.csv_path)
            # Normalize text columns so later string assignments do not hit float/object mismatches.
            for col in ['id', 'rating', 'notes', 'masked_path', 'clusters to combine']:
                if col in state.df.columns:
                    state.df[col] = state.df[col].fillna('').astype(str)
            if not all(col in state.df.columns for col in ['id', 'rating', 'notes', 'masked_path', 'clusters to combine']):
                print("Adding missing columns to existing CSV...")
                if 'clusters to combine' not in state.df.columns:
                    state.df['clusters to combine'] = state.df['id'].apply(lambda x: f"{x}")
                if 'masked_path' not in state.df.columns:
                    state.df['masked_path'] = state.df['id'].apply(lambda x: f"masked/cluster_{x}_masked.h5")
                state.df.to_csv(state.csv_path, index=False)
            data = {
                "id": state.df['id'],
                "rating": state.df['rating'],
                "notes": state.df['notes'],
                "masked_path": state.df['masked_path'],
                "clusters to combine": state.df['clusters to combine']
            }
        else:
            # Create a fresh table with all IDs pre-populated
            data = {
                "id": state.crop_ids,
                "rating": ["Unsure"] * len(state.crop_ids),
                "notes": [""] * len(state.crop_ids),
                "masked_path": [f"masked/cluster_{cid}_masked.h5" for cid in state.crop_ids],
                "clusters to combine": [[f"{cid}"] for cid in state.crop_ids]
            }
        state.df = pd.DataFrame(data)
        state.df.to_csv(state.csv_path, index=False)
            
        if state.crop_ids:
            state.crop_ids = [str(cid) for cid in state.crop_ids]

            refresh_id_list()
            sync_id_list_value(state.crop_ids[0])
            jump_id_input.value = state.crop_ids[0]
            
            state.current_idx = 0
            show_current_crop()
        else:
            print("No matching crops found in folder!")

    @magicgui(call_button="Toggle Cluster Labels")
    def toggle_labels():
        state.show_labels = not state.show_labels
        label_layers = ["Labels_XY", "Labels_XZ", "Labels_YZ", "Labels_3D"]
        for name in label_layers:
            if name in viewer.layers:
                viewer.layers[name].visible = state.show_labels
        print(f"Labels {'visible' if state.show_labels else 'hidden'}")

    def show_current_crop():
        cid = state.crop_ids[state.current_idx]
        
        # Look inside the directory and find a file that ends with the cluster pattern
        m_dir = state.root_path / "masked"
        m_path = next(m_dir.glob(f"*cluster_{cid}_masked.h5"), None)

        r_dir = state.root_path / "raw"
        r_path = next(r_dir.glob(f"*cluster_{cid}_raw.h5"), None)

        # Safety check to make sure the files were actually found
        if m_path is None or r_path is None:
            raise FileNotFoundError(f"Could not find masked or raw files for cluster {cid} with pattern *cluster_{cid}_*.h5")
        viewer.layers.clear()
 
        with h5py.File(m_path, 'r') as f:
            dat_shape = f[list(f.keys())[0]].shape

        if state.cluster_locs is not None and state.cluster_labels is not None:
            pts, ids = get_nearby_clusters(cid, dat_shape)
            if pts is not None and ids is not None:
                load_4panel_view((pts, ids), type="points")
        load_4panel_view(m_path, type="masked")
        load_4panel_view(r_path, type="raw")
        # Update the label on our custom button container
        status_label.value = f"Crop {state.current_idx + 1} of {len(state.crop_ids)} (ID: {cid})"

    # --- Navigation Widgets & Logic ---

    # 1. Search by ID (Type and hit Enter)
    jump_id_input = widgets.LineEdit(label="Jump to ID:", value="")
    
    @jump_id_input.changed.connect
    def jump_by_id_logic():
        target_id = str(jump_id_input.value).strip()
        if target_id in state.crop_ids:
            state.current_idx = state.crop_ids.index(target_id)
            # Update the list selection to match without triggering another load
            sync_id_list_value(target_id)
            show_current_crop()
        elif target_id != "":
            print(f"ID {target_id} not found in current folder.")
        else:
            print("Please enter a valid ID to jump to.")

    # 2. ID list (native Qt widget; reliable inside napari docks)
    id_list_widget = QListWidget()
    id_list_widget.setMinimumHeight(180)

    def refresh_id_list():
        id_list_widget.blockSignals(True)
        id_list_widget.clear()
        id_list_widget.addItems(state.crop_ids)
        id_list_widget.blockSignals(False)

    def sync_id_list_value(cid):
        text = str(cid)
        matches = id_list_widget.findItems(text, Qt.MatchExactly)
        if matches:
            id_list_widget.blockSignals(True)
            id_list_widget.setCurrentItem(matches[0])
            id_list_widget.blockSignals(False)

    def jump_by_id_list_logic(value):
        selected_id = str(value).strip()
        if selected_id and selected_id in state.crop_ids:
            new_idx = state.crop_ids.index(selected_id)
            if new_idx != state.current_idx:
                state.current_idx = new_idx
                jump_id_input.value = selected_id
                show_current_crop()

    # Connect via itemClicked to be compatible across Qt versions
    id_list_widget.itemClicked.connect(lambda item: jump_by_id_list_logic(item.text()))

    @magicgui(call_button="Next Crop >>")
    def next_crop():
        if state.crop_ids and state.current_idx < len(state.crop_ids) - 1:
            state.current_idx += 1
            # Sync UI
            next_id = state.crop_ids[state.current_idx]
            sync_id_list_value(next_id)
            jump_id_input.value = next_id
            show_current_crop()

    @magicgui(call_button="<< Prev Crop")
    def prev_crop():
        if state.crop_ids and state.current_idx > 0:
            state.current_idx -= 1
            # Sync UI
            prev_id = state.crop_ids[state.current_idx]
            sync_id_list_value(prev_id)
            jump_id_input.value = prev_id
            show_current_crop()

    status_label = widgets.Label(value="No folder loaded")

    # Combine all navigation into one container
    next_prev_widget = widgets.Container(
        widgets=[
            jump_id_input,
            widgets.Container(widgets=[prev_crop, next_crop], layout="horizontal", labels=False),
            status_label
        ],
        labels=True
    )

    @magicgui(call_button='Save Rating',
              notes={"label": "Notes:"},
              combine_ids={"label": "Combine with Clusters (comma-separated IDs):"})
    def rate_crop(rating = Rating.Unsure, notes = str(""), combine_ids = str("")) -> None:
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
        # if there are multiple IDs to combine, split by comma and strip whitespace, then join with semicolon for storage
        # if there are no IDs provided, just use the CID of the current crop
        ids_list = [cid.strip() for cid in combine_ids.split(',') if cid.strip()]
        ids_string = ";".join(ids_list)
        if not ids_string:
            ids_string = current_id
        state.df.loc[state.df['id'] == current_id, 'clusters to combine'] = ids_string
        
        # Save the whole thing back to CSV
        state.df.to_csv(state.csv_path, index=False)
        
        print(f"Updated ID {current_id} with {rating.name}")
        
        # Optional: Automatically move to next after rating
        next_crop()
 
    #viewer.window.add_dock_widget(load_crop, area='right')    
    #viewer.window.add_dock_widget(load_raw, area='right')

    viewer.window.add_dock_widget(folder_navigator, area='right', name="1. Setup")
    viewer.window.add_dock_widget(id_list_widget, area='right', name="2. ID List")
    viewer.window.add_dock_widget(next_prev_widget, area='right', name="3. Navigation")
    viewer.window.add_dock_widget(toggle_labels, area='right', name="4. Toggle Cluster Labels")
    viewer.window.add_dock_widget(load_crop, area='right', name="Optional: Load Individual Crop")
    viewer.window.add_dock_widget(rate_crop, area='right', name="5. Scoring")

    napari.run()