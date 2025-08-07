from sklearn.cluster import DBSCAN
import numpy as np
import zarr
import sys
import numpy as np

def cluster_vesicles(prediction_path, clusters_path="dbscan_clusters.npz"):
    """
    Cluster vesicles using DBSCAN and save the results.
    
    Parameters:
    - prediction_path: Path to the zarr file containing Hough transformed data. (e.g., '/home/predictions.zarr/predict/Prediction/Hough_transformed')
    - html_path: Path to save the HTML visualization of clusters.
    - clusters_path: Path to save the clustered coordinates and labels.
    """
    
    # ===== Load Data =====
    print("Opening Zarr file and streaming non-zero points...")
    z = zarr.open(prediction_path, mode='r')
    
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
    clustering = DBSCAN(eps=10, min_samples=200).fit(locs) # eps 87, min samples 100
    labels = clustering.labels_
    
    # ===== Save coordinates + labels =====
    np.savez_compressed(clusters_path, locs=locs, labels=labels)


if __name__ == "__main__":

    if len(sys.argv) < 2:
        print("Usage: python cluster_vesicles.py <prediction_path> [html_path] [clusters_path]")
        sys.exit(1)
    prediction_path = sys.argv[1]
    if len(sys.argv)==3:
        clusters_path = sys.argv[2]
    else:
        clusters_path = "dbscan_clusters.npz"

    cluster_vesicles(prediction_path, clusters_path)

