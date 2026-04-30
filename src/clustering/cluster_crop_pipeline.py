from cluster_vesicles import cluster_vesicles
from parallel_masking import mask_clusters_parallel
from crop_clusters_parallel import crop_clusters_parallel
from get_cluster_counts import count_positive_per_cluster
import os

def cluster_crop_pipeline(
    predictions_path,
    eps,
    min_samples,
    raw_path,
    dilation=2,
    n_jobs=4,
    chunk_size=100000,
    min_size=None,
    max_size=None,
    save_name_prefix=None,
):
    """
    Full pipeline to cluster vesicles, mask raw data, crop clusters, and count positives.
    1. Cluster vesicles using DBSCAN and save results to npz.
    2. Mask raw data based on clusters in parallel.
    3. Crop clusters from raw data in parallel.
    4. Count number of positive vesicles per cluster and save as CSV.
    Uses vindex for safe point-wise indexing in chunks.

    Inputs:
    - predictions_path: Path to hough transformed zarr data.
    - eps: DBSCAN clustering parameter eps, float.
    - min_samples: DBSCAN clustering parameter min_samples, int.
    - raw_path: Path to raw zarr dataset.
    - dilation: Dilation size for masking, int.
    - n_jobs: Number of parallel jobs for masking and cropping, int.
    - chunk_size: Chunk size for counting positives, int.
    - min_size: Minimum cluster size to include in counting, int or None.
    - max_size: Maximum cluster size to include in counting, int or None.
    - use_filenames: Whether to use filenames in CSV output, bool.

    Outputs:
    - npz file with cluster labels and locations.
    - Masked raw data saved as HDF5.
    - Cropped cluster data saved as HDF5.
    - CSV file with positive and negative counts per cluster.
    """

    if save_name_prefix:
        prefix = save_name_prefix
    else:
        prefix_path = predictions_path.split(".zarr")[0]
        prefix = prefix_path.split("/")[-1]
    npz_path = f"{prefix}_clusters_eps{eps}ms{min_samples}.npz"
    csv_out = f"{prefix}_cluster_counts.csv"
    out_masked_path = f"{prefix}_masked.h5"
    out_cropped_path = f"{prefix}_cropped.h5"

    # -------------------------- Step 1: Cluster vesicles and save to npz
    if os.path.exists(npz_path):
        recreate = input(f"Cluster file {npz_path} already exists, would you like to recreate it? (y/n): ")
    else:
        recreate = 'y'  # If file doesn't exist, we need to create it

    if recreate.lower() != 'y':
        print("Skipping clustering step.")
    else:
        print("Starting vesicle clustering...")
        cluster_vesicles(predictions_path, npz_path, eps, min_samples)

    print("Clustering completed. Starting masking and cropping...")

    # ------------------------- Masking raw data in parallel

    if os.path.exists(out_masked_path):
        recreate_mask = input(f"Masked file {out_masked_path} already exists, would you like to recreate it? (y/n): ")
    else:         
        recreate_mask = 'y'  # If file doesn't exist, we need to create it
        
    if recreate_mask.lower() != 'y':
        print("Skipping masking step.")
    else:
        print("Starting masking of raw data...")
        mask_clusters_parallel(
            raw_path,
            out_masked_path,
            npz_path,
            n_jobs=n_jobs,
            plot=True,
            dilation=dilation,
        )

        continue_pipeline = input("Do you want to continue the pipeline with the current mask data? (y/n): ")
        while continue_pipeline.lower() != 'y' and continue_pipeline.lower() != 'n':
            print("-----") 
            print("Invalid input. Please enter 'y' or 'n' only.")
            continue_pipeline = input("Do you want to continue the pipeline with the current mask data? (y/n): ")
        print("-----")

        if continue_pipeline.lower() == 'n':
            print("Pipeline stopped by user. You can rerun the pipeline with the generated masked data.")
            return

    print("Masking completed. Starting cropping...")

    # ------------------------- Crop clusters in parallel

    if os.path.exists(out_cropped_path):
        recreate_crop = input(f"Cropped file {out_cropped_path} already exists, would you like to recreate it? (y/n): ")
    else:
        recreate_crop = 'y'  # If file doesn't exist, we need to create it
    
    if recreate_crop.lower() != 'y':
        print("Skipping cropping step.")
    else:
        crop_clusters_parallel(
            raw_path,
            out_masked_path,
            npz_path,
            out_cropped_path,
            n_jobs=n_jobs,
        )

    print("Cropping completed. Starting counting positives...")

    # --------------------- Step 4: Count positive vesicles per cluster and save as CSV

    if os.path.exists(csv_out):
        recreate_csv = input(f"CSV file {csv_out} already exists, would you like to recreate it? (y/n): ")
    else:
        recreate_csv = 'y'  # If file doesn't exist, we need to create it
    if recreate_csv.lower() != 'y':
        print("Skipping counting positives step.")
    else:
        raw_path_for_counting = f"{raw_path}/raw" if raw_path.endswith("predict") else raw_path
        count_positive_per_cluster(
            npz_path,
            raw_path_for_counting,
            csv_out,
            chunk_size=chunk_size,
            min_size=min_size,
            max_size=max_size,
            use_filenames=True,
        )

    print("Clustering and cropping pipeline completed.")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Cluster and crop vesicles pipeline.")
    parser.add_argument("--predictions_path", required=True, help="Path to hough transformed zarr data.")
    parser.add_argument("--eps", required=False, help="DBscan clustering parameter eps, float.")
    parser.add_argument("--min_samples", required=False, help="DBscan clustering parameter min_samples.")
    parser.add_argument("--raw_path", required=True, help="Path to raw zarr dataset.")
    parser.add_argument("--dilation", type=int, default=2, help="Dilation size for masking.")
    parser.add_argument("--n_jobs", type=int, default=4, help="Number of parallel jobs.")
    parser.add_argument("--chunk_size", type=int, default=100000, help="Chunk size for counting positives.")
    parser.add_argument("--min_size", type=int, default=None, help="Minimum cluster size to include.")
    parser.add_argument("--max_size", type=int, default=None, help="Maximum cluster size to include.")
    parser.add_argument("--save_name_prefix", required=False, help="Prefix for saved files, instead of using the default based on predictions path.")

    args = parser.parse_args()
    print("Starting clustering and cropping pipeline...")

    cluster_crop_pipeline(
        predictions_path=args.predictions_path,
        eps=float(args.eps),
        min_samples=int(args.min_samples),
        raw_path=args.raw_path,
        dilation=args.dilation,
        n_jobs=args.n_jobs,
        chunk_size=args.chunk_size,
        min_size=args.min_size,
        max_size=args.max_size,
        use_filenames=args.use_filenames,
    )
# Example usage:
#
# python src/clustering/cluster_crop_pipeline.py \
    # --predictions_path /Users/administrator/Documents/CorrelatingNeuronalActivity/VesicleDetection/data/19-13_subvolume_0647-1670_6x6x6nm.zarr/predict/Predictions/31_07_2025/Hough_transformed \
    # --npz_path Users/administrator/Documents/CorrelatingNeuronalActivity/VesicleDetection/data/clusters/clusters_1913sbv_eps6ms200 \
    # --eps 6.0 \
    # --min_samples 200 \
    # --raw_path /Users/administrator/Documents/CorrelatingNeuronalActivity/VesicleDetection/data/19-13_subvolume_0647-1670_6x6x6nm.zarr/predict/ \
    # --dilation 2 \
    # --n_jobs 8 \
    # --chunk_size 100000 \
    # --min_size  200\
    # --max_size 10000 \
    # --use_filenames