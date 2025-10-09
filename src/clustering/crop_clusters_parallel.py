import os
import numpy as np
import zarr
from joblib import Parallel, delayed

def crop_volume(center, size, data, out_path, name):
    """Crop a cube of given size around center from data and save as zarr."""
    half = size // 2
    z, y, x = center

    # Bounds check — ensure full cube fits inside data
    if (
        z - half < 0 or z + half > data.shape[0] or
        y - half < 0 or y + half > data.shape[1] or
        x - half < 0 or x + half > data.shape[2]
    ):
        print(f"⚠️  Skipping crop at {center} (out of bounds)")
        return None  # Skip out-of-bounds crops


    # Crop bounds
    zmin, zmax = max(z - half, 0), min(z + half, data.shape[0])
    ymin, ymax = max(y - half, 0), min(y + half, data.shape[1])
    xmin, xmax = max(x - half, 0), min(x + half, data.shape[2])

    crop = data[zmin:zmax, ymin:ymax, xmin:xmax]

    # Sanity check — ensure cubic shape
    if not (crop.shape[0] == crop.shape[1] == crop.shape[2] == size):
        print(f"⚠️  Non-cubic crop at {center}, got shape {crop.shape}")
        return None

    # Save to zarr
    root = zarr.open(out_path, mode="w")
    root.create_dataset(
        name,
        shape=crop.shape,
        chunks=(32, 128, 128),
        dtype=crop.dtype,
        data=crop,
        overwrite=False,
    )
    return out_path


def process_cluster_crop(cid, locs, raw, labels, masked, crop_size_vox, out_dir):
    cluster_points = locs[labels == cid]
    if len(cluster_points) == 0:
        return None
    center = np.round(cluster_points.mean(axis=0)).astype(int)

    raw_path = os.path.join(out_dir, f"cluster_{cid}_raw.zarr")
    masked_path = os.path.join(out_dir, f"cluster_{cid}_masked.zarr")

    crop_volume(center, crop_size_vox, raw, raw_path, "raw")
    crop_volume(center, crop_size_vox, masked, masked_path, "masked")
    print(f"Saved cluster {cid} crops")
    return cid


def crop_clusters_parallel(raw_path, masked_path, npz_path=None, out_dir=None, crop_um=1.92, voxel_size_nm=(6,6,6), n_jobs=4, top_clusters=None, cluster_ids=None, min_points=None, max_points=None):
    # Load raw & masked datasets
    f_raw = zarr.open(raw_path, mode="r")
    raw = f_raw["raw"]

    f_masked = zarr.open(masked_path, mode="r")
    masked = f_masked["masked_raw"]

    # Load clusters
    data = np.load(npz_path)
    locs, labels = data["locs"], data["labels"]

    # Filter clusters
    unique, counts = np.unique(labels[labels >= 0], return_counts=True)

    if min_points is not None:
        mask = counts >= min_points
        unique, counts = unique[mask], counts[mask]

    if max_points is not None:
        mask = counts <= max_points
        unique, counts = unique[mask], counts[mask]

    if cluster_ids is not None:
        selected_clusters = [cid for cid in cluster_ids if cid in unique]
    elif top_clusters is not None:
        sorted_idx = np.argsort(-counts)[:top_clusters]
        selected_clusters = unique[sorted_idx]
    else:
        selected_clusters = unique

    print(f"Cropping {len(selected_clusters)} clusters")

    # Convert crop size to voxels
    crop_size_nm = crop_um * 1000  # µm → nm
    crop_size_vox = int(crop_size_nm / voxel_size_nm[0])  # assume isotropic 6nm voxels
    print(f"Crop size: {crop_size_vox} voxels ({crop_um} µm)")

    os.makedirs(out_dir, exist_ok=True)

    # Parallel crop
    Parallel(n_jobs=n_jobs, prefer="threads")(
        delayed(process_cluster_crop)(cid, locs, raw, labels, masked, crop_size_vox, out_dir)
        for cid in selected_clusters
    )

    print(f"All crops saved in {out_dir}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Crop raw and masked data around cluster centers")
    parser.add_argument("--raw_path", type=str, required=True, help="Path to raw zarr dataset, the folder containing raw")
    parser.add_argument("--masked_path", type=str, required=True, help="Path to masked zarr dataset, the folder containing masked_raw")
    parser.add_argument("--npz", type=str, required=True, help="Path to clusters npz")
    parser.add_argument("--out_dir", type=str, required=True, help="Output directory for crops")
    parser.add_argument("--n_jobs", type=int, default=4, help="Number of parallel workers")
    parser.add_argument("--crop_um", type=float, default=1.92, help="Crop size in micrometers")
    parser.add_argument("--top_clusters", type=int, default=None, help="If set, only process this many largest clusters")
    parser.add_argument("--cluster_ids", type=int, nargs='*', default=None, help="If set, only process these cluster IDs")
    parser.add_argument ("--min_points", type=int, default=500, help="Minimum points to consider a cluster")
    parser.add_argument("--max_points", type=int, default=10000, help="Maximum points to consider a cluster")

    args = parser.parse_args()
    crop_clusters_parallel(
        raw_path=args.raw_path,
        masked_path=args.masked_path,
        npz_path=args.npz,
        out_dir=args.out_dir,
        crop_um=args.crop_um,
        n_jobs=args.n_jobs,
        top_clusters=args.top_clusters or None,
        cluster_ids=args.cluster_ids or None,
        min_points=args.min_points or None,
        max_points=args.max_points or None,
    )

# Usage example:
#
# python crop_clusters.py \
#   --raw_path data/raw.zarr \
#   --masked_path data/masked.zarr \
#   --npz clusters.npz \
#   --out_dir crops_filtered \
#   --min_points 200 \
#   --max_points 2000 \
#   --n_jobs 8
#
#
# or:
#
# python src/clustering/crop_clusters_parallel.py \
# --raw_path /Users/administrator/Documents/CorrelatingNeuronalActivity/VesicleDetection/data/19-13_subvolume_0647-1670_6x6x6nm.zarr/predict/ \
# --masked_path /Users/administrator/Documents/CorrelatingNeuronalActivity/VesicleDetection/data/19-13_subvolume_0647-1670_6x6x6nm.zarr/predict/all_masked_dilation1_eps6ms60 \
# --npz /Users/administrator/Documents/CorrelatingNeuronalActivity/VesicleDetection/data/clusters/clusters_1913sbv_eps6ms60.npz \
# --out_dir /Users/administrator/Documents/CorrelatingNeuronalActivity/VesicleDetection/data/19-13_subvolume_0647-1670_6x6x6nm.zarr/cluster_crops/ \
# --n_jobs 8 \
# --min_points 2000 

