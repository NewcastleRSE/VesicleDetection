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

    # create save folder based on prefix    
    save_folder = f"data/{prefix}/"
    os.makedirs(save_folder, exist_ok=True)
    # save the input parameters to a text file in the save folder for reference
    with open(os.path.join(save_folder, "input_parameters.txt"), "w") as f:
        f.write(f"predictions_path: {predictions_path}\n")
        f.write(f"eps: {eps}\n")
        f.write(f"min_samples: {min_samples}\n")
        f.write(f"raw_path: {raw_path}\n")
        f.write(f"dilation: {dilation}\n")
        f.write(f"n_jobs: {n_jobs}\n")
        f.write(f"chunk_size: {chunk_size}\n")
        f.write(f"min_size: {min_size}\n")
        f.write(f"max_size: {max_size}\n")

    npz_path = f"{save_folder}{prefix}_clusters_eps{eps}ms{min_samples}.npz"
    csv_out = f"{save_folder}{prefix}_labels.csv"
    out_masked_path = f"{save_folder}{prefix}_masked"
    out_cropped_path = f"{save_folder}{prefix}_cropped"


    # -------------------------- Step 1: Cluster vesicles and save to npz
    if os.path.exists(npz_path):
        recreate = input(f"Cluster file {npz_path} already exists, would you like to recreate it? (y/n): ")
    else:
        recreate = 'y'  # If file doesn't exist, we need to create it

    if recreate.lower() != 'y':
        print("Skipping clustering step.")
    else:
        if os.path.exists(f"{predictions_path}/candidates.csv"):
            print("Starting vesicle clustering using the candidates file...")
            cluster_vesicles(f"{predictions_path}/candidates.csv", npz_path, eps, min_samples)
        else:
            print("Starting vesicle clustering using the hough transformed data...")
            cluster_vesicles(f"{predictions_path}/Hough_transformed", npz_path, eps, min_samples)

    print("Clustering completed. Starting masking and cropping...")

    # ------------------------- Masking raw data in parallel

    if os.path.exists(f"{out_masked_path}_convexhull"):
        recreate_mask = input(f"Masked file {out_masked_path}_convexhull already exists, would you like to recreate it? (y/n): ")
    else:         
        recreate_mask = 'y'  # If file doesn't exist, we need to create it
        
    if recreate_mask.lower() != 'y':
        print("Skipping masking step.")
    else:
        print("Starting masking of raw data...")
        mask_clusters_parallel(
            raw_path,
            f"{out_masked_path}_convexhull",
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
        
    print("Convex hull masking complete, starting vesicle masking...")

    
    if os.path.exists(f"{out_masked_path}_vesicle"):
        recreate_mask = input(f"Masked file {out_masked_path}_vesicle already exists, would you like to recreate it? (y/n): ")
    else:         
        recreate_mask = 'y'  # If file doesn't exist, we need to create it
        
    if recreate_mask.lower() != 'y':
        print("Skipping masking step.")
    else:
        print("Starting masking of raw data...")
        mask_clusters_parallel(
            raw_path,
            f"{out_masked_path}_vesicle",
            predictions_path,
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

    if os.path.exists(f"{out_cropped_path}_convexhull"):
        recreate_crop = input(f"Cropped file {out_cropped_path}_convexhull already exists, would you like to recreate it? (y/n): ")
    else:
        recreate_crop = 'y'  # If file doesn't exist, we need to create it
    
    if recreate_crop.lower() != 'y':
        print("Skipping convex hull cropping step.")
    else:
        crop_clusters_parallel(
            raw_path,
            f"{out_masked_path}_convexhull",
            npz_path,
            f"{out_cropped_path}_convexhull",
            n_jobs=n_jobs,
        )
    print("Convex hull cropping complete, starting vesicle cropping...")

    if os.path.exists(f"{out_cropped_path}_vesicle"):
        recreate_crop = input(f"Cropped file {out_cropped_path}_vesicle already exists, would you like to recreate it? (y/n): ")
    else:
        recreate_crop = 'y'  # If file doesn't exist, we need to create it
    if recreate_crop.lower() != 'y':
        print("Skipping vesicle masked cropping step.")
    else:
        crop_clusters_parallel(
            raw_path,
            f"{out_masked_path}_vesicle",
            npz_path,
            f"{out_cropped_path}_vesicle",
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
        
        count_positive_per_cluster(
            npz_path,
            f"{predictions_path}/Hough_transformed",
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
    parser.add_argument("--eps", required=True, help="DBscan clustering parameter eps, float.")
    parser.add_argument("--min_samples", required=True, help="DBscan clustering parameter min_samples.")
    parser.add_argument("--raw_path", required=True, help="Path to raw zarr dataset.")
    parser.add_argument("--dilation", type=int, default=2, help="Dilation size for masking.")
    parser.add_argument("--n_jobs", type=int, default=4, help="Number of parallel jobs.")
    parser.add_argument("--chunk_size", type=int, default=100000, help="Chunk size for counting positives.")
    parser.add_argument("--min_size", type=int, default=None, help="Minimum cluster size to include.")
    parser.add_argument("--max_size", type=int, default=None, help="Maximum cluster size to include.")
    parser.add_argument("--save_name_prefix", required=False, help="Prefix for saved files, instead of using the default based on predictions path.")

    args = parser.parse_args()
    print("Starting clustering and cropping pipeline...")

    if not os.path.exists(args.predictions_path):
        print(f"Error: Predictions path '{args.predictions_path}' does not exist.")
        exit(1)

    if not args.save_name_prefix:
        args.save_name_prefix = None

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
        save_name_prefix=args.save_name_prefix,
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