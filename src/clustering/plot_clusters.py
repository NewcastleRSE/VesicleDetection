import os
import zarr
import napari


def view_clusters_in_napari(path):
    """
    Load one or more saved cluster crops into napari.

    Parameters
    ----------
    path : str
        Path to either:
          - a directory containing multiple cluster subfolders
          - a single cluster folder itself
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Path does not exist: {path}")

    cluster_paths = []

    # Case 1: path is a folder containing multiple clusters
    if os.path.isdir(path):
        # Check if this looks like a cluster folder
        if os.path.exists(os.path.join(path, ".zarray")):
            # This is a single zarr cluster folder
            cluster_paths = [path]
        else:
            # Otherwise, assume it's a parent folder
            cluster_paths = [
                os.path.join(path, d)
                for d in os.listdir(path)
                if os.path.isdir(os.path.join(path, d)) and
                   os.path.exists(os.path.join(path, d, ".zarray"))
            ]
    else:
        raise ValueError(f"Expected directory path, got file: {path}")

    if not cluster_paths:
        raise RuntimeError(f"No cluster zarr datasets found in {path}")

    print(f"Found {len(cluster_paths)} clusters")

    # Open napari viewer
    viewer = napari.Viewer()

    for cluster_path in cluster_paths:
        cluster_name = os.path.basename(cluster_path)
        root = zarr.open(cluster_path, mode="r")

        if "raw" in root:
            viewer.add_image(root["raw"], name=f"{cluster_name}_raw")
        if "masked" in root:
            viewer.add_image(root["masked"], name=f"{cluster_name}_masked", opacity=0.6)

    napari.run()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="View cluster crops in napari")
    parser.add_argument("path", type=str, help="Path to cluster folder or parent directory")
    args = parser.parse_args()

    view_clusters_in_napari(args.path)