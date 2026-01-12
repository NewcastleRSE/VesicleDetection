import numpy as np
import plotly.express as px
from sklearn.cluster import DBSCAN
import zarr
import sys

# ===== Parameters =====
prediction_path = sys.argv[1]
if len(sys.argv)==4:
    html_path = sys.argv[2]
    clusters_path = sys.argv[3]
else:
    html_path = "dbscan_clusters.html"
    clusters_path = "dbscan_clusters.npz"

# ===== Load Data =====
f_prediction = zarr.open(prediction_path, mode='r')
hough_transformed = f_prediction[:]

# Get coordinates of non-zero points
locs = np.where(hough_transformed != 0)
locs = np.asarray(locs).T  # (z, y, x)

# ===== DBSCAN =====
clustering = DBSCAN(eps=87, min_samples=100).fit(locs)
labels = clustering.labels_

# ===== Save coordinates + labels =====
np.savez_compressed(clusters_path, locs=locs, labels=labels)

# ===== Interactive 3D Plot =====
fig = px.scatter_3d(
    x=locs[:, 2],  # X
    y=locs[:, 1],  # Y
    z=locs[:, 0],  # Z
    color=labels.astype(str),  # Convert labels to strings so Plotly treats them as categories
    symbol=np.where(labels == -1, "x", "circle"),  # Noise as 'x'
    opacity=0.7
)
fig.update_traces(marker=dict(size=3))
fig.update_layout(title="3D DBSCAN Clustering")

# Save interactive HTML
fig.write_html(html_path)

print(f"Interactive 3D plot saved to {html_path}")
print(f"Cluster data saved to {clusters_path}")

