import numpy as np
import zarr
import csv
from collections import defaultdict
from tqdm import tqdm

def count_positive_per_cluster(npz_path, prediction_path, csv_out, min_size=None, max_size=None, use_filenames=False, chunk_size=100_000):
    """
    Count number of positive vesicles per cluster and save as CSV.
    Uses vindex for safe point-wise indexing in chunks.
    """
    data = np.load(npz_path)
    locs = data["locs"]   # (N,3)
    labels = data["labels"]

    z = zarr.open(prediction_path, mode="r")

    total = defaultdict(int)
    positive = defaultdict(int)
    negative = defaultdict(int)
    whatisthis = defaultdict(list)

    n_points = len(locs)
    print(f"Processing {n_points} points in chunks of {chunk_size}...")

    for start in tqdm(range(0, n_points, chunk_size), desc="Counting positives"):
        end = min(start + chunk_size, n_points)
        locs_chunk = locs[start:end]
        labels_chunk = labels[start:end]

        # tuple of coordinate arrays
        locs_t = tuple(locs_chunk.T)

        # use point-wise indexing
        vals = z.vindex[locs_t]

        for lbl, val in zip(labels_chunk, vals):
            if lbl != -1 and val == 1:
                positive[lbl] += 1
            elif lbl != -1 and val == 2:
                negative[lbl] += 1  # count noise positives separately
            elif lbl != -1:
                whatisthis[lbl].append(val)


    # --- Compute cluster sizes for filtering ---
    unique, counts = np.unique(labels[labels >= 0], return_counts=True)
    cluster_sizes = dict(zip(unique, counts))

    # Determine which clusters to include
    valid_clusters = set(unique)
    if min_size is not None:
        valid_clusters = {cid for cid in valid_clusters if cluster_sizes[cid] >= int(min_size)}
    if max_size is not None:
        valid_clusters = {cid for cid in valid_clusters if cluster_sizes[cid] <= int(max_size)}

    print(f"Found {len(valid_clusters)} clusters within size range.")

    # Save results
    with open(csv_out, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["cluster", "positive_count", "negative_count", "cluster_size"])
        
        all_keys = sorted(valid_clusters)
        for cid in all_keys:
            cluster_name = f"cluster_{cid}_masked.zarr" if use_filenames else cid
            writer.writerow([cluster_name,
                            positive.get(cid, 0),
                            negative.get(cid, 0),
                            cluster_sizes.get(cid, 0)])
        # for cid, n in sorted(positive.items()):
        #     cluster_name = f"cluster_{cid}_masked.zarr" if use_filenames else cid
        #     writer.writerow([cluster_name, n])

    print(f"Saved positive counts to {csv_out}")
    return positive, negative, cluster_sizes, whatisthis

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 4:
        print("Usage: python get_cluster_counts.py <clusters.npz> <prediction.zarr> <output.csv> [--min_size N] [--max_size M] [--use_filenames]")
        sys.exit(1)
    
    npz_path = sys.argv[1]
    prediction_path = sys.argv[2]
    csv_out = sys.argv[3]
    min_size = sys.argv[sys.argv.index("--min_size") + 1] if "--min_size" in sys.argv else None
    max_size = sys.argv[sys.argv.index("--max_size") + 1] if "--max_size" in sys.argv else None

    pos,neg,tot,what = count_positive_per_cluster(npz_path, prediction_path, csv_out, min_size, max_size, use_filenames=False)
