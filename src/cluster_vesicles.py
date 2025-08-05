from sklearn.cluster import DBSCAN
import numpy as np
from src.visualisation import imshow_napari_prediction
import zarr
import sys
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import matplotlib.pyplot as plt

# need to open the zarr container and extract the hough transformed data
#prediction_path = sys.argv[1] # this needs to be the Hough transform path
prediction_path = '/Users/administrator/Documents/CorrelatingNeuronalActivity/VesicleDetection/data/19-13_sv0647-1670_ss377_519_crp467x_440y.zarr/predict/Predictions/01_07_2025/Hough_transformed'

f_prediction = zarr.open(prediction_path, mode='r')
hough_transformed = f_prediction[:]

locs = np.where(hough_transformed!=0)
locs = np.asarray(locs).T

clustering = DBSCAN(eps=18, min_samples=100).fit(locs)
labels = clustering.labels_  # cluster labels (-1 means noise)

# Unique cluster labels
unique_labels = set(labels)
colors = plt.cm.tab20(np.linspace(0, 1, len(unique_labels)))

# 3D plot
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

for label, color in zip(unique_labels, colors):
    mask = (labels == label)
    xyz = locs[mask]

    if label == -1:
        # Noise
        ax.scatter(xyz[:, 2], xyz[:, 1], xyz[:, 0], 
                   c='k', marker='x', s=10, alpha=0.5, label="Noise")
    else:
        ax.scatter(xyz[:, 2], xyz[:, 1], xyz[:, 0], 
                   c=[color], s=10, alpha=0.6, label=f"Cluster {label}")

ax.set_title("3D DBSCAN Clustering Results")
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")
ax.legend(loc="best")
plt.show()