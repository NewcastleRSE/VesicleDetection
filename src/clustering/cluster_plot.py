import argparse
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import napari
import zarr
from scipy.spatial import ConvexHull
import trimesh

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
    locs = data["locs"]
    labels = data["labels"]

    # Filter out noise points
    mask = labels != -1
    locs = locs[mask]
    labels = labels[mask]

    # Filter cluster IDs if given
    if cluster_ids is not None:
        print(f"Filtering clusters to only include: {cluster_ids}")
        cluster_ids = set(cluster_ids)
        mask = np.array([label in cluster_ids for label in labels])
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
        print(
            f"Limiting to top {top_clusters} clusters by size, out of {len(np.unique(labels))} total clusters"
        )
        unique, counts = np.unique(labels, return_counts=True)
        sorted_clusters = unique[np.argsort(-counts)]
        allowed = set(sorted_clusters[:top_clusters])
        mask = np.array([label in allowed for label in labels])
        locs = locs[mask]
        labels = labels[mask]

    return locs, labels


def crop_volumes(raw_data, hough_data, locs, crop_min=None, crop_max=None):
    if crop_min is None or crop_max is None:
        print("Computing crop bounds from cluster locations")
        min_coords = locs.min(axis=0)
        max_coords = locs.max(axis=0) + 1  # include max index
    else:
        min_coords = np.array(crop_min)
        max_coords = np.array(crop_max)
    print(
        f"Cropping to bounding box: Z[{min_coords[0]}:{max_coords[0]}], Y[{min_coords[1]}:{max_coords[1]}], X[{min_coords[2]}:{max_coords[2]}]"
    )

    raw_crop = raw_data[
        min_coords[0] : max_coords[0],
        min_coords[1] : max_coords[1],
        min_coords[2] : max_coords[2],
    ]

    hough_crop = None
    if hough_data is not None:
        hough_crop = hough_data[
            min_coords[0] : max_coords[0],
            min_coords[1] : max_coords[1],
            min_coords[2] : max_coords[2],
        ]

    locs_crop = locs - min_coords
    return raw_crop, hough_crop, locs_crop


def hull_to_mask(vertices, faces, shape, dilation=0):
    """
    Convert a convex hull mesh to a 3D binary mask.

    Parameters
    ----------
    vertices : (N, 3) array
        Coordinates of mesh vertices.
    faces : (M, 3) array
        Triangular faces (vertex indices).
    shape : tuple
        Shape of the output mask (z, y, x).
    dilation : int
        Optional dilation radius in voxels.

    Returns
    -------
    mask : ndarray (bool)
        Binary mask where True is inside the hull.
    """
    # Create trimesh object
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)

    # Voxelise at resolution of image (pitch = 1 assumes vertices are in voxel coords)
    vox = mesh.voxelized(pitch=1).fill()
    mask = vox.matrix.astype(bool)

    # Place into image space (ensure correct shape)
    img_mask = np.zeros(shape, dtype=bool)
    min_corner = np.floor(mesh.bounds[0]).astype(int)
    z, y, x = mask.shape
    img_mask[
        min_corner[0] : min_corner[0] + z,
        min_corner[1] : min_corner[1] + y,
        min_corner[2] : min_corner[2] + x,
    ] = mask

    if dilation > 0:
        from scipy.ndimage import binary_dilation

        img_mask = binary_dilation(img_mask, iterations=dilation)

    return img_mask


def crop_around_mask(raw_data, mask, voxel_size_nm=(6, 6, 6), crop_size_um=2.0):
    """
    Crop a cube around the mask centroid with padding to ensure full mask coverage.

    Returns raw crop, mask crop, and crop bounds.
    """
    crop_size_nm = crop_size_um * 1000
    crop_size_voxels = [int(crop_size_nm / vs) for vs in voxel_size_nm]
    half_crop = [s // 2 for s in crop_size_voxels]

    # centroid of the mask
    coords = np.argwhere(mask)
    cz, cy, cx = coords.mean(axis=0).astype(int)

    # initial bounding box
    zmin, zmax = cz - half_crop[0], cz + half_crop[0]
    ymin, ymax = cy - half_crop[1], cy + half_crop[1]
    xmin, xmax = cx - half_crop[2], cx + half_crop[2]

    # adjust to fully include mask
    mzmin, mymin, mxmin = coords.min(axis=0)
    mzmax, mymax, mxmax = coords.max(axis=0)
    zmin = min(zmin, mzmin)
    zmax = max(zmax, mzmax)
    ymin = min(ymin, mymin)
    ymax = max(ymax, mymax)
    xmin = min(xmin, mxmin)
    xmax = max(xmax, mxmax)

    # clip to data bounds
    zmin, ymin, xmin = np.maximum([zmin, ymin, xmin], 0)
    zmax, ymax, xmax = np.minimum([zmax, ymax, xmax], np.array(raw_data.shape))

    cropped_raw = raw_data[zmin:zmax, ymin:ymax, xmin:xmax]
    cropped_mask = mask[zmin:zmax, ymin:ymax, xmin:xmax]

    return cropped_mask, cropped_raw


# Plotting functions ------------------------------


def plot_points_matplotlib(locs, labels, output=None):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    unique_labels = np.unique(labels)
    for cid in unique_labels:
        pts = locs[labels == cid]
        ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], s=1, alpha=0.5)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    if output:
        plt.savefig(output)
    else:
        plt.show()


def plot_points_plotly(locs, labels, output=None):
    fig = go.Figure()
    unique_labels = np.unique(labels)
    for cid in unique_labels:
        pts = locs[labels == cid]
        fig.add_trace(
            go.Scatter3d(
                x=pts[:, 0],
                y=pts[:, 1],
                z=pts[:, 2],
                mode="markers",
                marker=dict(size=1, opacity=0.3),
            )
        )
    fig.update_layout(scene=dict(aspectmode="data"))
    if output:
        fig.write_html(output)
    else:
        fig.show()


def napari_plot(
    raw_data,
    hough_data,
    locs,
    labels,
    out_dir,
    napari_plot_types=None,
    dilation=None,
    create_crops=False,
):
    # Get a colormap with enough distinct colors
    unique_labels = np.unique(labels)
    print(f"Unique labels found: {len(unique_labels)}")

    napari_plot_types = (
        napari_plot_types.split(",")
        if isinstance(napari_plot_types, str)
        else napari_plot_types
    )
    print(f"Napari plot types: {napari_plot_types}")

    viewer = napari.Viewer()
    viewer.add_image(raw_data, name="Raw")
    if hough_data is not None:
        viewer.add_image(hough_data, name="Hough_transformed", opacity=0.3)

    if (
        napari_plot_types == "points"
        or isinstance(napari_plot_types, list)
        and "points" in napari_plot_types
    ):
        point_features = {
            "clusters": labels,
            "colours": labels.astype(float) / max(labels),
        }  # Use labels as point features for color mapping
        viewer.add_points(
            locs,
            size=1,
            features=point_features,
            face_color="colours",
            face_colormap="viridis",
            name="Cluster points",
        )

    if (
        napari_plot_types == "volume"
        or isinstance(napari_plot_types, list)
        and "volume" in napari_plot_types
    ):
        # Prepare labeled volume
        label_vol = np.zeros(raw_data.shape, dtype=np.int32)
        for cid in np.unique(labels):
            points = locs[labels == cid]
            label_vol[
                points[:, 0].astype(int),
                points[:, 1].astype(int),
                points[:, 2].astype(int),
            ] = cid + 1
        viewer.add_labels(label_vol, name="Cluster volumes")

    if (
        napari_plot_types == "mask"
        or isinstance(napari_plot_types, list)
        and "mask" in napari_plot_types
    ):
        use_mask = True
    else:
        use_mask = False

    if (
        napari_plot_types == "shapes"
        or use_mask
        or isinstance(napari_plot_types, list)
        and "shapes" in napari_plot_types
    ):
        raw_masked = np.copy(raw_data)
        hulls = calculate_cluster_hulls(locs, labels)
        print(f"Plotting {len(hulls)} convex hulls")
        for points, vertices, faces, cid in hulls:
            print(
                f"Processing cluster {cid} with {len(points)} points and {len(faces)} faces"
            )
            # 1. Extract unique vertices from hull.simplices
            unique_vertex_indices = np.unique(faces.flatten())
            vertices_coords = points[unique_vertex_indices]
            if dilation is not None:
                centroid = vertices_coords.mean(axis=0)
                vertices_coords = centroid + dilation * (vertices_coords - centroid)
            # 2. Remap simplices to new vertex indices
            index_map = {
                old_idx: new_idx
                for new_idx, old_idx in enumerate(unique_vertex_indices)
            }
            faces_remapped = np.array([[index_map[i] for i in face] for face in faces])
            values = np.ones(vertices_coords.shape[0]) * cid
            if isinstance(napari_plot_types, list) and "shapes" in napari_plot_types:
                viewer.add_surface(
                    (vertices_coords, faces_remapped, values),
                    name=f"Cluster {cid}",
                )
            if use_mask:
                mask = hull_to_mask(points, faces, raw_data.shape, dilation=3)
                raw_masked[mask] = 0
        if use_mask:
            root = zarr.open(out_dir, mode="w")
            root.create_dataset(
                "masked_raw",
                shape=raw_masked.shape,
                data=raw_masked,
                chunks=(32, 128, 128),
                overwrite=True,
            )
            viewer.add_image(raw_masked, name="Masked Clusters", opacity=0.5)
        for (
            cid
        ) in hulls:  # moving to ensure that all masks are created before any crops
            if create_crops:
                cropped_mask, cropped_raw = crop_around_mask(raw_masked, mask)
                crop_name = f"{out_dir}/cluster_{cid}_crop"
                root = zarr.open(crop_name, mode="w")
                root.create_dataset(
                    "raw",
                    shape=cropped_raw.shape,
                    data=cropped_raw,
                    chunks=(32, 128, 128),
                    overwrite=True,
                )
                root.create_dataset(
                    "mask",
                    shape=cropped_mask.shape,
                    data=cropped_mask,
                    chunks=(32, 128, 128),
                    overwrite=True,
                )
                # viewer.add_image(cropped_raw, name=f'Raw Crop {cid}')
                # viewer.add_image(cropped_mask.astype(np.float32), name=f'Mask Crop {cid}', opacity=0.5)
                print(f"Saved cropped data for cluster {cid} to {crop_name}")

    napari.run()


def cluster_plotter():
    parser = argparse.ArgumentParser(
        description="Cluster visualization tool with napari and plotting"
    )
    parser.add_argument("--data_path", type=str, help="Path to raw data zarr container")
    parser.add_argument(
        "--prediction_path", type=str, help="Path to prediction zarr container"
    )
    parser.add_argument("--npz", type=str, help="Path to npz file with locs and labels")
    parser.add_argument(
        "--downsample", type=int, default=1, help="Downsample factor for points"
    )
    parser.add_argument(
        "--top_clusters", type=int, default=None, help="Number of top clusters to show"
    )
    parser.add_argument(
        "--clusters", nargs="*", type=int, help="Specific cluster ids to plot"
    )
    parser.add_argument(
        "--napari_plot",
        nargs="*",
        type=str,
        help="Choose napari plot type, either points, volume, spheres, shapes or mask. Can have multiple options. e.g. --napari_plot points volume",
    )
    parser.add_argument("--html", type=str, help="Path to save plotly HTML")
    parser.add_argument("--png", type=str, help="Path to save matplotlib PNG")
    parser.add_argument(
        "--crop_coords",
        nargs="*",
        help="Pass cropping coordinates to enforce a crop of the data, in the form zmin,zmax,ymin,ymax,xmin,xmax. E.g. --crop_coords 10,20,30,40,50,60",
    )
    parser.add_argument(
        "--dilation", type=float, default=None, help="Dilation factor for shapes"
    )
    parser.add_argument(
        "--create_crops",
        action="store_true",
        help="If set, will create cropped volumes and save to masked_clusters in data_path",
    )

    args = parser.parse_args()

    # Load files -----------------------------
    if args.npz:
        data = np.load(args.npz)
        locs = data["locs"]
        labels = data["labels"]
    else:
        raise RuntimeError("Must provide npz file with cluster locs and labels")

    if args.data_path:
        f_data = zarr.open(args.data_path, mode="r")
        raw_data = f_data["raw"][:]
        out_dir = f"{args.data_path}masked_clusters"
    else:
        raw_data = None

    if args.prediction_path:
        f_pred = zarr.open(args.prediction_path, mode="r")
        hough_data = f_pred["Hough_transformed"]
        if hough_data is not None:
            hough_data = hough_data[:]
    else:
        hough_data = None

    # Filter clusters -----------------------------
    locs, labels = load_clusters(
        args.npz,
        downsample=args.downsample,
        cluster_ids=args.clusters,
        top_clusters=args.top_clusters,
    )

    # Crop volumes -----------------------------
    if args.crop_coords:
        crop_coords = list(map(int, args.crop_coords.split(",")))
        if len(crop_coords) != 6:
            raise ValueError(
                "Crop coordinates must be in the form zmin,zmax,ymin,ymax,xmin,xmax"
            )
        zmin, zmax, ymin, ymax, xmin, xmax = crop_coords
        raw_data, hough_data, locs = crop_volumes(
            raw_data,
            hough_data,
            locs,
            crop_min=(zmin, ymin, xmin),
            crop_max=(zmax, ymax, xmax),
        )
    else:
        raw_data, hough_data, locs = crop_volumes(raw_data, hough_data, locs)

    # Plotting -----------------------------
    if args.html:
        plot_points_plotly(locs, labels, output=args.html)

    if args.png:
        plot_points_matplotlib(locs, labels, output=args.png)

    if not args.prediction_path:
        args.prediction_path = None

    if args.dilation:
        dilation = args.dilation
    else:
        dilation = None

    if args.napari_plot:
        if raw_data is None:
            raise RuntimeError("Raw data is required for napari plotting")
        napari_plot(
            raw_data,
            hough_data,
            locs,
            labels,
            out_dir,
            args.napari_plot,
            dilation,
            create_crops=args.create_crops,
        )


if __name__ == "__main__":
    cluster_plotter()
