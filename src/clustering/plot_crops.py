import os
import zarr
import napari
from collections import defaultdict


def open_first_array(zarr_path):
    """Open a .zarr path and return the first array inside."""
    root = zarr.open(zarr_path, mode="r")
    if isinstance(root, zarr.Array):
        return root
    elif isinstance(root, zarr.Group):
        # try common dataset names
        for key in ["raw", "mask", "masked_raw"]:
            if key in root:
                return root[key]
        # fallback: first array in group
        for key, val in root.items():
            if isinstance(val, zarr.Array):
                return val
    raise ValueError(f"No array dataset found in {zarr_path}")


def view_clusters_in_napari(path, subset=None):
    if os.path.isdir(path) and not path.endswith(".zarr"):
        files = [f for f in os.listdir(path) if f.endswith(".zarr")]
    else:
        files = [os.path.basename(path)]
        path = os.path.dirname(path)

    cluster_pairs = defaultdict(dict)
    for fname in files:
        if "_raw" in fname:
            cid = fname.split("_raw")[0].split("cluster_")[-1]
            cluster_pairs[cid]["raw"] = os.path.join(path, fname)
        elif "_masked" in fname:
            cid = fname.split("_masked")[0].split("cluster_")[-1]
            cluster_pairs[cid]["masked"] = os.path.join(path, fname)

    cluster_ids = sorted(cluster_pairs.keys(), key=lambda x: int(x))
    if subset is not None:
        cluster_ids = cluster_ids[:subset]

    viewer = napari.Viewer()

    for cid in cluster_ids:
        paths = cluster_pairs[cid]
        if "raw" in paths:
            raw = open_first_array(paths["raw"])
            viewer.add_image(raw, name=f"cluster_{cid}_raw")

        if "masked" in paths:
            masked = open_first_array(paths["masked"])
            viewer.add_image(masked, name=f"cluster_{cid}_masked", opacity=0.6)

    napari.run()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="View cluster crops in napari")
    parser.add_argument(
        "path", type=str, help="Path to cluster folder or parent directory"
    )
    parser.add_argument(
        "--subset", type=int, default=None, help="If set, only view this many clusters"
    )
    args = parser.parse_args()

    view_clusters_in_napari(args.path, subset=args.subset)
