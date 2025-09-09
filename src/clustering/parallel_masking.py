from cluster_plot import load_clusters
from joblib import Parallel, delayed
from scipy.ndimage import binary_dilation
import numpy as np
from scipy.spatial import ConvexHull, Delaunay
import zarr
import napari

def hull_to_mask_delaunay(points, shape, mask_zarr, dilation=0):
    """Write the convex hull of points into mask_zarr inplace."""
    hull = ConvexHull(points)
    delaunay = Delaunay(points[hull.vertices])

    # Compute bounding box of cluster
    mins = np.floor(points.min(axis=0)).astype(int)
    maxs = np.ceil(points.max(axis=0)).astype(int) + 1

    # Clip to image bounds
    mins = np.maximum(mins, 0)
    maxs = np.minimum(maxs, shape)

    # Generate coordinates only inside bounding box
    zz, yy, xx = np.indices((maxs - mins))
    coords = np.c_[zz.ravel() + mins[0],
                   yy.ravel() + mins[1],
                   xx.ravel() + mins[2]]

    inside = delaunay.find_simplex(coords) >= 0
    mask_local = inside.reshape((maxs - mins))

    if dilation > 0:
        mask_local = binary_dilation(mask_local, iterations=dilation)

    # Write into global mask dataset
    mask_zarr[mins[0]:maxs[0], mins[1]:maxs[1], mins[2]:maxs[2]] |= mask_local


def process_cluster(locs, labels, cid, mask_zarr, dilation=0):
    """Worker: compute cluster mask and OR into global mask_zarr."""
    cluster_points = locs[labels == cid]
    if len(cluster_points) < 4:
        print(f"Cluster {cid} too small, skipping.")
        return
    hull_to_mask_delaunay(cluster_points, mask_zarr.shape, mask_zarr, dilation=dilation)
    print(f"Cluster {cid} processed.")


def mask_clusters_parallel(raw_path, out_path, npz, plot=False, n_jobs=4, dilation=0):
    # Open raw data
    f_raw = zarr.open(raw_path, mode='r')
    raw = f_raw['raw']

    locs, labels = load_clusters(npz)

    # Create output zarr datasets
    root_out = zarr.open(out_path, mode='w')
    masked = root_out.create_dataset(
        "masked_raw",
        shape=raw.shape,
        chunks=raw.chunks,
        dtype=raw.dtype,
        overwrite=True,
    )
    mask = root_out.create_dataset(
        "mask",
        shape=raw.shape,
        chunks=raw.chunks,
        dtype="bool",
        overwrite=True,
    )

    # Run clusters in parallel
    unique_clusters = np.unique(labels[labels >= 0])
    Parallel(n_jobs=n_jobs, prefer="processes")(
        delayed(process_cluster)(locs, labels, cid, mask, dilation)
        for cid in unique_clusters
    )

    # Apply mask once
    coords = np.where(mask)
    for z in np.unique(coords[0]):
        # load slice from both mask and data
        z_mask = mask[z, :, :]
        z_data = raw[z, :, :]

        # apply mask in memory
        z_data[z_mask] = 0

        # write back
        masked[z, :, :] = z_data
    print(f"Masked raw saved at {out_path}")

        # Plot in napari
    if plot:
        viewer = napari.Viewer()
        viewer.add_image(raw, name="Raw")
        viewer.add_image(masked, name="Masked Raw")
        viewer.add_labels(mask.astype(np.uint8), name="Mask", opacity=0.5)
        napari.run()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        description="Mask clusters in raw data using convex hulls."
    )
    parser.add_argument("raw_path", type=str, help="Path to input zarr with raw data.")
    parser.add_argument("out_path", type=str, help="Path to output zarr for masked data and mask.")
    parser.add_argument("npz", type=str, help="Path to npz file with locs and labels.")
    parser.add_argument("--n_jobs", type=int, default=4, help="Number of parallel jobs.")
    parser.add_argument("--plot", action="store_true", help="Whether to plot results in napari.")
    parser.add_argument("--dilation", type=int, default=0, help="Dilation radius for mask.")
    args = parser.parse_args()

    mask_clusters_parallel(raw_path=args.raw_path,
                           out_path=args.out_path,
                           npz=args.npz,
                           n_jobs=args.n_jobs,
                           plot=args.plot,
                           dilation=args.dilation or 0)
    
    # useage:
    #
    # python src/clustering/parallel_masking.py <raw_zarr_path> <output_zarr_path> <localizations_npy_path> <labels_npy_path> --n_jobs 8 --plot
    # eg.
    # python src/clustering/parallel_masking.py /Users/administrator/Documents/CorrelatingNeuronalActivity/VesicleDetection/data/19-13_subvolume_0647-1670_6x6x6nm.zarr/predict /Users/administrator/Documents/CorrelatingNeuronalActivity/VesicleDetection/data/19-13_subvolume_0647-1670_6x6x6nm.zarr/predict/all_masked /path/to/locs.npy /path/to/labels.npy --n_jobs 8 --plot