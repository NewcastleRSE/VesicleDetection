import argparse
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import plotly.graph_objects as go
import napari
import zarr
from napari.layers import Shapes
from scipy.spatial import ConvexHull

# Helper functions ------------------------------

def calculate_cluster_hulls(locs, labels):
    hulls = []
    for cid in np.unique(labels):
        cluster_points = locs[labels == cid]
        if len(cluster_points) < 4:
            # ConvexHull needs at least 4 non-coplanar points in 3D
            continue
        try:
            hull = ConvexHull(cluster_points)
            hulls.append((cluster_points, hull.vertices, hull.simplices, cid))
        except Exception as e:
            print(f"Skipping cluster {cid} due to ConvexHull error: {e}")
    return hulls


def load_clusters(npz_path, downsample=1.0, cluster_ids=None, top_clusters=None):
    data = np.load(npz_path)
    locs = data['locs']
    labels = data['labels']

    # Filter out noise points
    mask = labels != -1
    locs = locs[mask]
    labels = labels[mask]

    # Filter cluster IDs if given
    if cluster_ids is not None:
        print(f"Filtering clusters to only include: {cluster_ids}")
        cluster_ids = set(cluster_ids)
        mask = np.array([l in cluster_ids for l in labels])
        locs = locs[mask]
        labels = labels[mask]

    # Downsample points if requested
    if downsample < 1.0:
        print(f"Downsampling points by factor: {downsample}")
        keep = np.random.rand(len(locs)) < downsample
        locs = locs[keep]
        labels = labels[keep]

    # Limit to top clusters by number of points if requested
    if top_clusters is not None:
        print(f"Limiting to top {top_clusters} clusters by size")
        unique, counts = np.unique(labels, return_counts=True)
        sorted_clusters = unique[np.argsort(-counts)]
        allowed = set(sorted_clusters[:top_clusters])
        mask = np.array([l in allowed for l in labels])
        locs = locs[mask]
        labels = labels[mask]

    return locs, labels

def crop_volumes(raw_data, hough_data, locs, crop_min=None, crop_max=None):
    if crop_min is None or crop_max is None:
        print("Computing crop bounds from cluster locations")
        min_coords = locs.min(axis=0)
        max_coords = locs.max(axis=0) +1  # include max index
    else:
        min_coords = np.array(crop_min)
        max_coords = np.array(crop_max)
    print(f"Cropping to bounding box: Z[{min_coords[0]}:{max_coords[0]}], Y[{min_coords[1]}:{max_coords[1]}], X[{min_coords[2]}:{max_coords[2]}]")

    raw_crop = raw_data[min_coords[0]:max_coords[0],
                        min_coords[1]:max_coords[1],
                        min_coords[2]:max_coords[2]]

    hough_crop = None
    if hough_data is not None:
        hough_crop = hough_data[min_coords[0]:max_coords[0],
                               min_coords[1]:max_coords[1],
                               min_coords[2]:max_coords[2]]

    locs_crop = locs - min_coords
    return raw_crop, hough_crop, locs_crop

def compute_cluster_spheres(locs, labels, cluster_ids=None):
    print("computing spheres for clusters")
    spheres = []
    unique_labels = np.unique(labels)
    if cluster_ids is not None:
        unique_labels = [cid for cid in unique_labels if cid in cluster_ids]

    for cid in unique_labels:
        points = locs[labels == cid]
        if len(points) == 0:
            continue
        center = points.mean(axis=0)
        dists = np.linalg.norm(points - center, axis=1)
        radius = dists.max()
        spheres.append((center[0], center[1], center[2], radius, cid))

    return np.array(spheres)

# Plotting functions ------------------------------

def plot_spheres_matplotlib(spheres, output=None):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    for cx, cy, cz, r in spheres:
        ax.scatter(cx, cy, cz, s=max(r*50, 1), alpha=0.5)  # size scales with radius
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    if output:
        plt.savefig(output)
    else:
        plt.show()

def plot_points_matplotlib(locs, labels, output=None):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    unique_labels = np.unique(labels)
    for cid in unique_labels:
        pts = locs[labels == cid]
        ax.scatter(pts[:,0], pts[:,1], pts[:,2], s=1, alpha=0.5)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    if output:
        plt.savefig(output)
    else:
        plt.show()

def plot_spheres_plotly(spheres, output=None):
    fig = go.Figure()
    for cx, cy, cz, r in spheres:
        fig.add_trace(go.Scatter3d(
            x=[cx], y=[cy], z=[cz],
            mode='markers',
            marker=dict(size=r*10, color='blue', opacity=0.5)
        ))
    fig.update_layout(scene=dict(aspectmode='data'))
    if output:
        fig.write_html(output)
    else:
        fig.show()

def plot_points_plotly(locs, labels, output=None):
    fig = go.Figure()
    unique_labels = np.unique(labels)
    for cid in unique_labels:
        pts = locs[labels == cid]
        fig.add_trace(go.Scatter3d(
            x=pts[:,0], y=pts[:,1], z=pts[:,2],
            mode='markers',
            marker=dict(size=1, opacity=0.3)
        ))
    fig.update_layout(scene=dict(aspectmode='data'))
    if output:
        fig.write_html(output)
    else:
        fig.show()

def napari_plot(raw_data, hough_data, locs, labels, napari_plot_types=None, spheres=None):
    # Get a colormap with enough distinct colors
    unique_labels = np.unique(labels)
    print(f"Unique labels found: {len(unique_labels)}")
    cmap = plt.get_cmap('tab20', len(unique_labels))
    label_to_color = {label: cmap(i) for i, label in enumerate(unique_labels)}

    napari_plot_types = napari_plot_types.split(',') if isinstance(napari_plot_types, str) else napari_plot_types
    print(f"Napari plot types: {napari_plot_types}")

    viewer = napari.Viewer()
    viewer.add_image(raw_data, name='Raw')
    if hough_data is not None:
        viewer.add_image(hough_data, name='Hough_transformed', opacity=0.3)

    if napari_plot_types=='points' or isinstance(napari_plot_types, list) and 'points' in napari_plot_types:
        point_features = {'clusters': labels,
                      'colours': labels.astype(float)/max(labels)}  # Use labels as point features for color mapping
        viewer.add_points(locs, size=1, features=point_features, face_color='colours', face_colormap='viridis', name='Cluster points')

    if napari_plot_types=='volume' or isinstance(napari_plot_types, list) and 'volume' in napari_plot_types:
        # Prepare labeled volume
        label_vol = np.zeros(raw_data.shape, dtype=np.int32)
        for cid in np.unique(labels):
            points = locs[labels == cid]
            label_vol[points[:,0].astype(int), points[:,1].astype(int), points[:,2].astype(int)] = cid + 1
        viewer.add_labels(label_vol, name='Cluster volumes')

    if napari_plot_types=='spheres' or isinstance(napari_plot_types, list) and 'spheres' in napari_plot_types:
        if spheres is None:
            print("No spheres data provided, skipping spheres plot")
            return
        shapes_data = []
        edge_colors = []
        face_colors = []
        print(f"Plotting {len(spheres)} spheres with labels {unique_labels}")

        for cx, cy, cz, r, label in spheres:
            circle = np.array([
                [cx + r*np.cos(t), cy + r*np.sin(t), cz]
                for t in np.linspace(0, 2*np.pi, 30)
            ])
            shapes_data.append(circle)
            print(f"Sphere at ({cx}, {cy}, {cz}) with radius {r} and label {label}")
            face_colors.append(label_to_color[label])
            print(f"Color for label {label}: {label_to_color[label]}")
            edge_colors.append(label_to_color[label])

        viewer.add_shapes(
            shapes_data,
            shape_type='polygon',
            edge_color=edge_colors,
            face_color="red",
            opacity=0.5,
            name='Cluster spheres'
        )

    if napari_plot_types=='shapes' or isinstance(napari_plot_types, list) and 'shapes' in napari_plot_types:
        hulls = calculate_cluster_hulls(locs, labels)
        print(f"Plotting {len(hulls)} convex hulls")
        for points, vertices, faces, cid in hulls:
            print("Vertices coords range:", points.min(axis=0), points.max(axis=0))
            print("Faces shape:", faces.shape)
            # 1. Extract unique vertices from hull.simplices
            unique_vertex_indices = np.unique(faces.flatten())
            vertices_coords = points[unique_vertex_indices]
            # 2. Remap simplices to new vertex indices
            index_map = {old_idx: new_idx for new_idx, old_idx in enumerate(unique_vertex_indices)}
            faces_remapped = np.array([[index_map[i] for i in face] for face in faces])
            values = np.ones(vertices_coords.shape[0]) * cid
            viewer.add_surface((vertices_coords, faces_remapped, values),
                            name=f'Cluster {cid}',)

    napari.run()



def cluster_plotter():
    parser = argparse.ArgumentParser(description="Cluster visualization tool with napari and plotting")
    parser.add_argument('--data_path', type=str, help='Path to raw data zarr container')
    parser.add_argument('--prediction_path', type=str, help='Path to prediction zarr container')
    parser.add_argument('--npz', type=str, help='Path to npz file with locs and labels')
    parser.add_argument('--downsample', type=int, default=1, help='Downsample factor for points')
    parser.add_argument('--top_clusters', type=int, default=None, help='Number of top clusters to show')
    parser.add_argument('--clusters', nargs='*', type=int, help='Specific cluster ids to plot')
    parser.add_argument('--napari_plot', nargs='*', type=str, help='Choose napari plot type, either points, volume, spheres or shapes. Can have multiple options. e.g. --napari_plot points volume')
    parser.add_argument('--spheres', action='store_true', help='Plot with whole cluster spheres view')
    parser.add_argument('--html', type=str, help='Path to save plotly HTML')
    parser.add_argument('--png', type=str, help='Path to save matplotlib PNG')
    parser.add_argument('--crop_coords', nargs='*', help='Pass cropping coordinates to enforce a crop of the data, in the form zmin,zmax,ymin,ymax,xmin,xmax. E.g. --crop_coords 10,20,30,40,50,60')   

    args = parser.parse_args()

    # Load files -----------------------------
    if args.npz:
        data = np.load(args.npz)
        locs = data['locs']
        labels = data['labels']
    else:
        raise RuntimeError("Must provide npz file with cluster locs and labels")

    if args.data_path:
        f_data = zarr.open(args.data_path, mode='r')
        raw_data = f_data['raw'][:]
    else:
        raw_data = None

    if args.prediction_path:
        f_pred = zarr.open(args.prediction_path, mode='r')
        hough_data = f_pred.get('Hough_transformed', None)
        if hough_data is not None:
            hough_data = hough_data[:]
    else:
        hough_data = None

    # Filter clusters -----------------------------
    locs, labels = load_clusters(args.npz, downsample=args.downsample, cluster_ids=args.clusters, top_clusters=args.top_clusters)

    # Crop volumes -----------------------------
    if args.crop_coords:
        crop_coords = list(map(int, args.crop_coords.split(',')))
        if len(crop_coords) != 6:
            raise ValueError("Crop coordinates must be in the form zmin,zmax,ymin,ymax,xmin,xmax")
        zmin, zmax, ymin, ymax, xmin, xmax = crop_coords
        raw_data, hough_data, locs = crop_volumes(raw_data, hough_data, locs,
                                                      crop_min=(zmin, ymin, xmin),
                                                      crop_max=(zmax, ymax, xmax))
    else:
        raw_data, hough_data, locs = crop_volumes(raw_data, hough_data, locs)
    

    # Compute spheres if requested -----------------------------
    spheres = compute_cluster_spheres(locs, labels) if args.spheres else None

    # Plotting -----------------------------
    if args.html:
        if args.spheres:
            plot_spheres_plotly(spheres, output=args.html)
        else:
            plot_points_plotly(locs, labels, output=args.html)
    
    if args.png:
        if args.spheres:
            plot_spheres_matplotlib(spheres, output=args.png)
        else:
            plot_points_matplotlib(locs, labels, output=args.png)
    
    if not args.prediction_path:
            args.prediction_path = None

    if args.napari_plot:
        if raw_data is None:
            raise RuntimeError("Raw data is required for napari plotting")
        napari_plot(raw_data, hough_data, locs, labels, args.napari_plot, spheres)

if __name__ == "__main__":
    cluster_plotter()


