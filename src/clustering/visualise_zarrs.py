import zarr
import napari
import sys
import h5py
from pathlib import Path
import numpy as np


def plot_files(paths):
    """Plot multiple zarr or h5 files in napari.

    Args:
        paths str: List of paths to zarr or h5 files.
    """
    viewer = napari.Viewer()

    for path in paths:
        print(f"Visualizing {path}")
        try:  # if path is a cluster id:
            int(path)  # should thow error if not int
            print(f"Interpreting {path} as cluster ID")
            plot_clusterid(path, viewer)
        except ValueError:
            if path.endswith(".h5"):
                print("Loading h5 file")
                f = h5py.File(path, "r")
                # Assume dataset name is same as file name without extension
                dset_name = path.split("/")[-1].replace(".h5", "")
                if dset_name in f:
                    dat = f[dset_name][:]
                    option = ""
                else:
                    # If not found, just take the first dataset
                    first_key = list(f.keys())[0]
                    dat = f[first_key][:]
                    option = ""
                f.close()
                viewer.add_image(
                    data=dat,
                    name=path.split("/")[-1] + option,
                    blending="additive",
                    colormap="grey",
                    contrast_limits=[dat.min(), dat.max()],
                )
            elif has_masked_and_raw(path):
                print("Loading raw and masked datasets from subfolders")
                # check if the path has subfolders 'raw' and 'masked'
                plot_all_cluster_crops(path, viewer)
            else:
                print("Loading zarr file")
                try:
                    image = zarr.open(path, mode="r")
                    if "raw" in image:
                        zarrdat = image["raw"][:, :, :]
                        option = "_raw"
                    elif "masked_raw" in image:
                        zarrdat = image["masked_raw"][:, :, :]
                        option = "_masked"
                    elif "mask" in image:
                        zarrdat = image["mask"][:, :, :]
                        option = "_mask"
                    elif "masked" in image:
                        zarrdat = image["masked"][:, :, :]
                        option = "_masked"
                    elif "masks" in image:
                        zarrdat = image["masks"][:, :, :]
                        option = "_masks"
                    else:
                        zarrdat = image[:, :, :]
                        option = ""
                    viewer.add_image(
                        data=zarrdat,
                        name=path.split("/")[-1] + option,
                        blending="additive",
                        colormap="grey",
                        contrast_limits=[zarrdat.min(), zarrdat.max()],
                    )
                except Exception as e:
                    print(f"Error loading {path} as zarr: {e}")
                    pass

    napari.run()


def plot_clusterid(path, viewer):
    """Plot cluster crops for a given cluster ID in napari.

    Args:
        cluster_id int: Cluster ID to plot.
        viewer napari.Viewer: Napari viewer instance.
    """
    # This function would implement logic to find and plot crops for the given cluster ID
    # For now, it's a placeholder
    print(f"Plotting crops for cluster ID: {path}")
    # Example: load crops from a predefined directory structure
    # and add them to the viewer
    mask_path = f"data/19-13_subvolume_0647-1670_6x6x6nm_cluster_crops/masked/cluster_{path}_masked.h5"
    raw_path = f"data/19-13_subvolume_0647-1670_6x6x6nm_cluster_crops/raw/cluster_{path}_raw.h5"
    print(f"Loading cluster mask from {mask_path}")
    f = h5py.File(mask_path, "r")
    dset_name = "masked"
    if dset_name in f:
        maskdat = f[dset_name][:]
    else:
        # If not found, just take the first dataset
        first_key = list(f.keys())[0]
        maskdat = f[first_key][:]
    f.close()
    print(f"Loading cluster raw from {raw_path}")
    f = h5py.File(raw_path, "r")
    dset_name = "raw"
    if dset_name in f:
        rawdat = f[dset_name][:]
    else:
        # If not found, just take the first dataset
        first_key = list(f.keys())[0]
        rawdat = f[first_key][:]
    f.close()
    center = np.array(rawdat.shape) / 2
    points = np.array([center])
    viewer.add_points(points, name="center marker", size=10, face_color="red")
    viewer.add_image(
        data=maskdat,
        name=f"cluster_{path}_masked",
        blending="additive",
        colormap="grey",
        contrast_limits=[maskdat.min(), maskdat.max()],
    )
    viewer.add_image(
        data=rawdat,
        name=f"cluster_{path}_masked",
        blending="additive",
        colormap="grey",
        contrast_limits=[rawdat.min(), rawdat.max()],
    )


def plot_all_cluster_crops(path, viewer):
    print("Loading raw and masked datasets")
    raw_path = Path(path) / "raw"
    masked_path = Path(path) / "masked"
    for crop_file in raw_path.glob("*.h5"):
        print(f"Loading raw crop from {crop_file}")
        f = h5py.File(crop_file, "r")
        dset_name = crop_file.stem  # dataset name is file name without extension
        if dset_name in f:
            rawdat = f[dset_name][:]
            option = "_raw"
            viewer.add_image(
                data=rawdat,
                name=crop_file.stem + option,
                blending="additive",
                colormap="grey",
                contrast_limits=[rawdat.min(), rawdat.max()],
            )
        f.close()
    for crop_file in masked_path.glob("*.h5"):
        print(f"Loading masked crop from {crop_file}")
        f = h5py.File(crop_file, "r")
        dset_name = crop_file.stem  # dataset name is file name without extension
        if dset_name in f:
            maskdat = f[dset_name][:]
            option = "_masked"
            viewer.add_image(
                data=maskdat,
                name=crop_file.stem + option,
                blending="additive",
                colormap="grey",
                contrast_limits=[maskdat.min(), maskdat.max()],
            )
        f.close()


def has_masked_and_raw(path):
    p = Path(path)
    return (p / "raw").is_dir() and (p / "masked").is_dir()


if __name__ == "__main__":
    if len(sys.argv) < 1:
        print(
            "Usage: python visualise_zarrs.py <zarr_path> <zarr_path1> <zarr_path2> ..."
        )
        sys.exit(1)
    else:
        paths = sys.argv[1:]

    plot_files(paths)
