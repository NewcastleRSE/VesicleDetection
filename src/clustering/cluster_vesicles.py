from sklearn.cluster import DBSCAN
import numpy as np
import zarr
import sys


def cluster_vesicles(
    prediction_path, clusters_path="dbscan_clusters.npz", eps=5, min_samples=100
):
    """
    Cluster vesicles using DBSCAN and save the results.

    Parameters:
    - prediction_path: Path to the zarr file containing Hough transformed data. (e.g., '/home/predictions.zarr/predict/Prediction/Hough_transformed')
    - clusters_path: Path to save the clustered coordinates and labels.
    """

    print("running clustering with parameters: ")
    print(f"eps: {eps}, min_samples: {min_samples}")

    # ===== Load Data =====
    print("Opening Zarr file and streaming non-zero points...")
    z = zarr.open(prediction_path, mode="r")

    # Memory-efficient way to extract all non-zero coords
    coords = []
    for index, val in np.ndenumerate(z):
        if val != 0:
            coords.append(index)

    locs = np.array(coords, dtype=np.int32)
    del coords  # cleanup

    print(f"Found {len(locs)} non-zero points")
    # hough_transformed = zarr.open(prediction_path, mode='r')
    # hough_transformed = hough_transformed[:]

    # # Get coordinates of non-zero points
    # locs = np.where(hough_transformed != 0)
    # del hough_transformed
    # locs = np.asarray(locs).T

    # ===== DBSCAN =====
    clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(locs)
    labels = clustering.labels_

    print(f"DBSCAN found {len(set(labels)) - (1 if -1 in labels else 0)} clusters")
    print("saving results to " + clusters_path)
    # ===== Save coordinates + labels =====
    np.savez_compressed(clusters_path, locs=locs, labels=labels)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(
            "Usage: python cluster_vesicles.py <prediction_path> [clusters_path] [eps] [min_samples]"
        )
        sys.exit(1)
    prediction_path = sys.argv[1]
    if len(sys.argv) == 3:
        clusters_path = sys.argv[2]
    elif len(sys.argv) > 3:
        clusters_path = (
            sys.argv[2] + "eps" + str(sys.argv[3]) + "ms" + str(sys.argv[4]) + ".npz"
        )
        eps = float(sys.argv[3])
        min_samples = int(sys.argv[4])
    else:
        clusters_path = "dbscan_clusters_eps5ms100.npz"
        eps = 5
        min_samples = 100

    cluster_vesicles(prediction_path, clusters_path, eps, min_samples)
