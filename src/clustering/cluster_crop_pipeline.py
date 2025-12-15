from cluster_vesicles import cluster_vesicles
from parallel_masking import mask_clusters_parallel
from crop_clusters_parallel import crop_clusters_parallel
from get_cluster_counts import count_positive_per_cluster

def cluster_crop_pipeline(
    predictions_path,
    npz_path,
    eps,
    min_samples,
    raw_path,
    out_masked_path,
    out_cropped_path,
    csv_out,
    dilation=2,
    n_jobs=4,
    chunk_size=100000,
    min_size=None,
    max_size=None,
    use_filenames=True,
):
    """
    Full pipeline to cluster vesicles, mask raw data, crop clusters, and count positives.
    """
    # Step 1: Cluster vesicles and save to npz
    print("Starting vesicle clustering...")
    cluster_vesicles(predictions_path, npz_path, eps, min_samples)

    print("Clustering completed. Starting masking and cropping...")
    # Step 2: Mask raw data in parallel
    mask_clusters_parallel(
        raw_path,
        out_masked_path,
        npz_path,
        n_jobs=n_jobs,
        plot=False,
        dilation=dilation,
    )

    print("Masking completed. Starting cropping...")
    # Step 3: Crop clusters in parallel
    crop_clusters_parallel(
        raw_path,
        out_masked_path,
        npz_path,
        out_cropped_path,
        n_jobs=n_jobs,
    )

    print("Cropping completed. Starting counting positives...")
    # Step 4: Count positive vesicles per cluster and save as CSV
    count_positive_per_cluster(
        npz_path,
        raw_path,
        csv_out,
        chunk_size=chunk_size,
        min_size=min_size,
        max_size=max_size,
        use_filenames=use_filenames,
    )

    print("Clustering and cropping pipeline completed.")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Cluster and crop vesicles pipeline.")
    parser.add_argument("--predictions_path", required=True, help="Path to hough transformed zarr data.")
    parser.add_argument("--npz_path", required=False, help="Output path for clustered npz file.")
    parser.add_argument("--eps", required=False, help="DBscan clustering parameter eps, float.")
    parser.add_argument("--min_samples", required=False, help="DBscan clustering parameter min_samples.")
    parser.add_argument("--raw_path", required=True, help="Path to raw zarr dataset.")
    parser.add_argument("--out_masked_path", required=True, help="Output path for masked raw data.")
    parser.add_argument("--out_cropped_path", required=True, help="Output path for cropped clusters.")
    parser.add_argument("--csv_out", required=True, help="Output CSV file for cluster counts.")
    parser.add_argument("--dilation", type=int, default=2, help="Dilation size for masking.")
    parser.add_argument("--n_jobs", type=int, default=4, help="Number of parallel jobs.")
    parser.add_argument("--chunk_size", type=int, default=100000, help="Chunk size for counting positives.")
    parser.add_argument("--min_size", type=int, default=None, help="Minimum cluster size to include.")
    parser.add_argument("--max_size", type=int, default=None, help="Maximum cluster size to include.")
    parser.add_argument("--use_filenames", action="store_true", help="Use filenames in CSV output.")

    args = parser.parse_args()
    print("Starting clustering and cropping pipeline...")

    cluster_crop_pipeline(
        predictions_path=args.predictions_path,
        npz_path=args.npz_path,
        eps=float(args.eps),
        min_samples=int(args.min_samples),
        raw_path=args.raw_path,
        out_masked_path=args.out_masked_path,
        out_cropped_path=args.out_cropped_path,
        csv_out=args.csv_out,
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
    # --out_masked_path /Users/administrator/Documents/CorrelatingNeuronalActivity/VesicleDetection/data/19-13_subvolume_0647-1670_6x6x6nm.zarr/masked_e6ms200 \
    # --out_cropped_path /Users/administrator/Documents/CorrelatingNeuronalActivity/VesicleDetection/data/cropped_clusters_e6ms200/ \
    # --csv_out data/cluster_counts.csv \
    # --dilation 2 \
    # --n_jobs 8 \
    # --chunk_size 100000 \
    # --min_size  2000\
    # --max_size 100000 \
    # --use_filenames