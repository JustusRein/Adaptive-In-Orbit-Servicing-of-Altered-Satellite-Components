import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt
import pathlib
from pathlib import Path
import datetime
import logging
import yaml
import cv2
import itertools

import random
import copy
from math import acos, degrees
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
from collections import Counter
from shapely.geometry import Point, Polygon
from shapely.ops import unary_union
from mpl_toolkits.axes_grid1 import make_axes_locatable
from grasp_store import GraspStore, GraspData

from shapely.validation import explain_validity, make_valid   # Shapely>=2.0
from shapely import set_precision
from shapely.errors import GEOSException
from shapely.geometry import MultiPolygon



#**************** path catalog **************

pcd_path = Path("Object/assembly_scan.ply")

gripper_path = Path("gripper_parameter/Franka.yaml")

npz_load_path = Path("")



#***************** Code parameters **********

plane_angle_thresh = 5.0
offset_thresh = 2.0

min_remaining_points = 100
min_points_per_plane = 100 # small plane filter threshold (80)

distance_threshold = 0.1     # plane fitting error
max_planes = 200              # maximum number of planes to detect (can be increased)

margin_points_between_planes = 1

tilt_symbol_start_dist = 18
tilt_symbol_handle_length = 15
tilt_symbol_finger_width_half = 5
tilt_symbol_finger_end_length = 8

plt_graphic_padding = 10
contour_image_padding = 10

spacing_edge_normal_factor = 1/3

gss_w1 = 0.5
gss_w2 = 0.5

roughness_tolerance = 1.0 # compensate surface roughness of real pointcloud

#********* image console ****************
no_image = True # running without image compute
save_image = False # running with image compute and save image

essential_image_only = False
no_skip = False
show_all_planes_and_normals = False
show_planes_parallel_clustering = False
show_plane_pairs = False
show_plane_pair_and_proj_in_pcd = True
show_proj_pts_p1 = True
show_proj_pts_p2 = True
show_proj_pts_p3 = True
show_proj_pts_p4 = True
show_proj_pts_p5 = True
show_P2345_in_pcd = True
show_each_P_in_pcd = False
show_plt_contour_P2_2d = True
show_P2_contour_3d = False
show_plt_all_tcp_grids = False
show_plt_TCP_each_edge = False
show_plt_bounding_boxes = False
show_plt_contours_Px_2d = True
show_P_contour_3d = False
show_feasible_each_edge = False
show_all_feasbile_in_2d = True
show_feasible_with_P_and_pcd = False

if no_image:
    no_skip = True


if essential_image_only:
    show_all_planes_and_normals = True
    show_planes_parallel_clustering = True
    show_plane_pairs = False
    show_plane_pair_and_proj_in_pcd = True
    show_proj_pts_p1 = True
    show_proj_pts_p2 = True
    show_proj_pts_p3 = True
    show_proj_pts_p4 = True
    show_proj_pts_p5 = True
    show_P2345_in_pcd = True
    show_each_P_in_pcd = False
    show_plt_contour_P2_2d = True
    show_P2_contour_3d = True
    show_plt_all_tcp_grids = True
    show_plt_TCP_each_edge = False 
    show_plt_bounding_boxes = False
    show_plt_contours_Px_2d = True
    show_P_contour_3d = True
    show_feasible_each_edge = False
    show_all_feasbile_in_2d = True
    show_feasible_with_P_and_pcd = True


#**************** path complement **************
pcd_name =  pcd_path.stem
gripper_name =  gripper_path.stem

now = datetime.datetime.now()
time_str = now.strftime("%Y%m%d_%H%M%S")  #  20251101_002345

results_basepath = Path("results") / f"{time_str}-{gripper_name}-{pcd_name}-{distance_threshold}"
graspdata_basepath = results_basepath / "grasp_data"
imageout_basepath = results_basepath / "image"

graspdata_basepath.mkdir(parents=True, exist_ok=True)
imageout_basepath.mkdir(parents=True, exist_ok=True)

npz_filename = f"{time_str}-{gripper_name}-{pcd_name}-{distance_threshold}.npz"
csv_filename = f"{time_str}-{gripper_name}-{pcd_name}-{distance_threshold}.csv"

npz_filename2 = f"candidate.npy"


npz_outpath = graspdata_basepath / npz_filename
csv_outpath = graspdata_basepath / csv_filename

npz_outpath2 = graspdata_basepath / npz_filename2





#*******************************************



# MM_TO_M = 0.001

# def load_yaml_in_m(filepath):
#     """Load YAML fle，change unit from 'mm' to 'm' """
#     with open(filepath, "r", encoding="utf-8") as f:
#         data_mm = yaml.safe_load(f)
    
#     
#     data_m = {
#         key: (value * MM_TO_M if isinstance(value, (int, float)) else value)
#         for key, value in data_mm.items()
#     }
#     return data_m

# # mm to m
# params = load_yaml_in_m(gripper_path)

with open(gripper_path, "r", encoding="utf-8") as f:
    params = yaml.safe_load(f)


a_pg = params["a_pg"] # Finger width
w_pg = params["w_pg"] # Internal Safespace Finger width 
v_pg = params["v_pg"] # External Safespace Finger width 
f_pg = params["f_pg"] # Distance gripper open
g_pg = params["g_pg"] # Distance gripper close
h_pg = params["h_pg"] # Gripper base bottom width
k_pg = params["k_pg"] # Safespace Gripper base bottom width 
q_pg = params["q_pg"] # Gripper base top width
r_pg = params["r_pg"] # Safespace Gripper base top width

y_pg = max(q_pg + 2*r_pg, h_pg + 2*k_pg, f_pg + 2*(a_pg + v_pg)) # Gripper Bounding box max width

b_pg = params["b_pg"] # Gripper area length end
c_pg = params["c_pg"] # Gripper area to (Safety space of Gripper)length end
d_pg = params["d_pg"] # Safespace Gripper length
x_pg = params["x_pg"] # Safespace Gripper end to rubber
n_pg = d_pg + c_pg + b_pg # Finger length
t_pg = params["t_pg"] # Gripper base bottom length
u_pg = params["u_pg"] # Gripper base top length
s_pg = n_pg + t_pg + u_pg + x_pg # Total gripper length

e_pg = params["e_pg"] # Finger depth
i_pg = params["i_pg"] # Safespace finger depth

z_pg = params["z_pg"] # Gripper area depth

l_pg = params["l_pg"] # Gripper base bottom depth
m_pg = params["m_pg"] # Safespace gripper base bottom depth
o_pg = params["o_pg"] # Gripper base top  depth
p_pg = params["p_pg"] # Safespace gripper base top depth

j_pg = max(l_pg + 2*m_pg, o_pg + 2*p_pg, e_pg + 2*i_pg)  # Gripper Bounding box max depth


ra = params["ra"] #width of last robot arm limb
rb = params["rb"] #depth of last robot arm limb
rc = params["rc"] #length of last robot arm limb
rd = max(ra,rb) #maximum diameter of last robot arm limb
re = params["re"] #robot arm diameter clearance
rf = params["rf"] #robot arm length clearance
rj = params["rj"] #repeatability of robot arm

print("Gripper parameters:")
print(f"a_pg: {a_pg:.3f} m")
print(f"w_pg: {w_pg:.3f} m")




# **************************** Aid Functions **************************************
def filter_by_normal_orientation(
    pcd, n_ref, cos_th=0.965, knn=30, radius=None, max_nn=50
):
    N = len(pcd.points)
    if N == 0:
        return pcd, np.zeros(0, dtype=bool)

    n_ref = np.asarray(n_ref, dtype=float).reshape(3)
    nr = np.linalg.norm(n_ref)
    if nr == 0 or not np.isfinite(nr):
        raise ValueError("n_ref must be a non-zero, finite vector")
    n_ref = n_ref / nr

    need_est = (not pcd.has_normals()) or (len(pcd.normals) != N)

    if need_est:
        if N < 3:
            # Not enough points to estimate with neighborhood PCA, directly assign the known plane normal vector
            pcd.normals = o3d.utility.Vector3dVector(np.repeat(n_ref[None, :], N, axis=0))
        else:
            # Estimate normals using neighborhood PCA
            if radius is None:
                pts = np.asarray(pcd.points)
                diag = float(np.linalg.norm(pts.max(0) - pts.min(0)))
                radius = max(1e-9, 0.02 * diag)  # Experience points: ~2% of the diagonal
            pcd.estimate_normals(
                o3d.geometry.KDTreeSearchParamHybrid(
                    radius=radius,
                    max_nn=min(max_nn, max(3, N-1))
                )
            )
            # Check again, if it still doesn't match, proceed with the assignment.
            if (not pcd.has_normals()) or (len(pcd.normals) != N):
                pcd.normals = o3d.utility.Vector3dVector(np.repeat(n_ref[None, :], N, axis=0))

    # Standardize directionality; default to direct assignment upon failure
    try:
        pcd.orient_normals_to_align_with_direction(n_ref)
    except RuntimeError:
        pcd.normals = o3d.utility.Vector3dVector(np.repeat(n_ref[None, :], N, axis=0))

    pcd.normalize_normals()

    normals = np.asarray(pcd.normals)
    cosang = np.clip(normals @ n_ref, -1.0, 1.0)
    # If already aligned, abs is not strictly required here; keeping abs is more stable.
    mask = (np.abs(cosang) >= float(cos_th))
    out = pcd.select_by_index(np.where(mask)[0])
    return out, mask

def remove_pcd_outlier_statistical(pcd, neighbots=20,std_ratio=1.0):

    if len(pcd.points) == 0:
        print("Remove Outlier Error: Input point cloud is empty.")
        return pcd,None
    
    #statistical
    if isinstance(pcd, o3d.geometry.PointCloud):
        filtered,ind = pcd.remove_statistical_outlier(nb_neighbors=neighbots, std_ratio=std_ratio)
        return filtered,ind
    elif isinstance(pcd,np.ndarray):
        pcloud = o3d.geometry.PointCloud()
        pcloud.points = o3d.utility.Vector3dVector(pcd)
        filtered,ind = pcloud.remove_statistical_outlier(nb_neighbors=neighbots, std_ratio=std_ratio)
        return np.asarray(filtered.points),ind
    else:
        print("Error: Input type is not supported, neither 'PointCloud' nor 'np.ndarray'.")
        return None,None
    

def remove_pcd_outlier_dbscan(pcd, eps=0.007, min_samples=20,min_cluster_ratio=0.02,verbose=True):
    if len(pcd.points) <= 500:
        pcd_null = o3d.geometry.PointCloud()
        pcd_null.points = o3d.utility.Vector3dVector([])
        return pcd_null,None
    else:
        return pcd,None


# **************************** Step 1: Read point cloud + Estimate outward normals ****************************

logging.basicConfig(level=logging.INFO, format="%(message)s")


def orient_normals_outward(
    pcd: o3d.geometry.PointCloud,
    radius_factor: float = 4.0,   # Normal search radius = radius_factor * avg_nn_distance
    main_eps_deg: float = 20.0,   # DBSCAN clustering angle threshold (degrees)
    min_samples: int = 20,
    k_consistency: int = 30,
):
    """Estimate and unify outward-facing normals"""
    if pcd.is_empty():
        raise ValueError("Point cloud is empty!")

    # Compute average neighbor distance as the scale
    dists = pcd.compute_nearest_neighbor_distance()
    avg_d  = np.mean(dists)
    n_radius = radius_factor * avg_d

    logging.info(f"Avg NN distance: {avg_d:.4f}, using normal radius {n_radius:.4f}")

    # 1) Estimate normals
    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=n_radius, max_nn=50)
    )
    pcd.normalize_normals()

    # 2) Find the main normal direction (optional)
    normals = np.asarray(pcd.normals)
    clustering = DBSCAN(
        eps=np.sin(np.deg2rad(main_eps_deg)) * 2,  # Approximation of cosine distance threshold
        min_samples=min_samples, metric="cosine"
    ).fit(normals)
    labels = clustering.labels_
    valid   = labels >= 0
    if not valid.any():
        logging.warning("DBSCAN found no clusters, skipping global normal flipping")
    else:
        counts  = Counter(labels[valid])
        main_lab = counts.most_common(1)[0][0]
        main_dir = normals[labels == main_lab].mean(0)
        main_dir /= np.linalg.norm(main_dir)
        # Vectorized flipping of normals
        dot = (normals * main_dir).sum(1)
        normals[dot < 0] *= -1
        pcd.normals = o3d.utility.Vector3dVector(normals)

    # 3) Local normal consistency (Open3D built-in)
    pcd.orient_normals_consistent_tangent_plane(k_consistency)
    return pcd


path = pathlib.Path(pcd_path)
pcd = o3d.io.read_point_cloud(str(path))

def pcd_scaler(pcd): # Scale the pcd in meters up to 1000 times.
    dists = pcd.compute_nearest_neighbor_distance()
    avg_d  = np.mean(dists)
    obb = pcd.get_oriented_bounding_box()
    dimensions = max(obb.extent)

    logging.info(f"Avg NN distance: {avg_d:.4f}")
    logging.info(f"Dimensions: {dimensions:.4f}")

    if dimensions <= 1:
        logging.info(f"Target pcd is measured in meters, and will be scaled 1000 times up for unit consistency.")
        return 1000
    else:
        logging.info(f"Target pcd is measured in mini meters, and won't be scaled up.")
        return 1

pcd.scale(pcd_scaler(pcd), pcd.get_center())
pcd.paint_uniform_color([0.6, 0.6, 0.6])
# dists = pcd.compute_nearest_neighbor_distance()
# avg_d  = np.mean(dists)
# if avg_d > 0.5:
#     scale_factor = 1/1000
#     pcd.scale(scale_factor, pcd.get_center())
# scale_factor = 1/1000
# pcd.scale(scale_factor, pcd.get_center())
# pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.01, max_nn=50))
# orient_normals_outward(pcd)
# o3d.visualization.draw_geometries([pcd], point_show_normal=True,window_name="Estimated external normal")
# ************************
original_points = np.asarray(pcd.points)
original_indices = np.arange(len(original_points))
# ---------------- Step 2: Initialization ----------------
plane_indices_list = []   # Store original indices of each detected plane
plane_colors = []         # Colors for visualization
plane_models = []         # Plane model parameters
plane_normals = []        # Plane normal vectors



# pcd,_ = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
# pcd = pcd.voxel_down_sample(voxel_size=0.01)
rest_pcd = copy.deepcopy(pcd)

# ---------------- Step 3: extract planes ----------------


def correct_normal_direction_by_density(pcd, plane_indices, plane_normal):
        
        all_normals = np.asarray(pcd.normals)
        outer_avg = all_normals[plane_indices].mean(axis=0)
        outer_avg /= np.linalg.norm(outer_avg)

        dot = np.dot(outer_avg, plane_normal)
        angle = np.arccos(np.clip(dot, -1.0, 1.0)) * 180 / np.pi
        print(f"average point normal vs fitted normal angle: {angle:.2f}°")
        
        # Compare with the originally fitted plane normal to determine whether flipping is needed
        if angle > 90 :
            plane_normal = -plane_normal  # Make it point outward
        
        return plane_normal

for i in range(max_planes):

    print(f"loop{i}")

    if len(rest_pcd.points) < min_remaining_points:
        print("Not enough points left to extract more planes.")
        break  # Too few points left, stop extracting planes

    plane_model, inliers = rest_pcd.segment_plane(
        distance_threshold=distance_threshold,
        ransac_n=3,
        num_iterations=1000)

    if len(inliers) < min_points_per_plane:
        print(f"Plane has only {len(inliers)} points, skipping.")
        break  # Plane too small, stop extracting

    # Save indices of the current plane (indices from the remaining PCD, mapped back to original PCD)
    current_points = np.asarray(rest_pcd.points)
    current_indices = np.arange(len(current_points))
    original_idx = original_indices[inliers]
    plane_indices_list.append(original_idx)



# ***************************************************
    #Save plane model and plane normal
    plane_models.append(plane_model) 
    normal_vector = np.asarray(plane_model[0:3])
    normal_vector = normal_vector / np.linalg.norm(normal_vector)
    # normal_vector = correct_normal_direction_by_density(pcd, original_idx, normal_vector)
    plane_normals.append(normal_vector)


    color = [random.random(), random.random(), random.random()]
    plane_colors.append(color)

    # Remaining point cloud
    rest_pcd = rest_pcd.select_by_index(inliers, invert=True)
    original_indices = np.delete(original_indices, inliers)



#******************* Merge detected planes ****************************

def _normalize_plane(a, b, c, d):
    """normalize the plane equation ax+by+cz+d=0 to ||[a,b,c]|| = 1,return (n_hat, d_hat)"""
    n = np.array([a, b, c], dtype=float)
    norm = np.linalg.norm(n)
    if norm == 0:
        raise ValueError("Invalid plane normal with zero length.")
    n_hat = n / norm
    d_hat = d / norm
    return n_hat, d_hat

def _angle_deg(n1, n2):

    cosv = float(np.clip(np.dot(n1, n2), -1.0, 1.0)) #Angle between two unit normals (in degrees).
    return np.degrees(np.arccos(cosv))

def _refit_plane_from_points(points_xyz):
    """
    Fit the best plane to a set of points using least squares (SVD). 
    Returns (plane_model[a,b,c,d], unit_normal)
    """
    P = np.asarray(points_xyz, dtype=float)
    if len(P) < 3:
        raise ValueError("Need at least 3 points to fit a plane.")
    centroid = P.mean(axis=0)
    Q = P - centroid
    # SVD: Normal vector corresponding to the smallest eigenvector
    _, _, vt = np.linalg.svd(Q, full_matrices=False)
    normal = vt[-1, :]
    normal /= np.linalg.norm(normal)
    d = -np.dot(normal, centroid)
    return np.array([normal[0], normal[1], normal[2], d], dtype=float), normal

# -----------------------
# Main function: merge coplanar planes
# -----------------------
def merge_coplanar_planes(
    plane_models,           # [ [a,b,c,d], ... ] possibly not normalized
    plane_normals,          # [ unit n_i, ... ] 
    plane_indices_list,     # [ idx_list_i, ... ] indices of each plane in the original point cloud
    all_points_xyz,         # np.asarray(pcd.points)
    angle_thresh_deg=5.0,   # Normal angle threshold (degrees)
    offset_thresh=1.0       # Offset difference threshold |d1 - d2| after aligning normal directions (same units as PCD coordinates)
):
    """
    return:
      new_plane_models, new_normals, new_indices_list
    """
    # Normalize plane models to unit normals and align their directions with existing plane_normals to avoid inconsistency
    planes = []
    for (model, n_unit, idxs) in zip(plane_models, plane_normals, plane_indices_list):
        a, b, c, d = model
        n_hat, d_hat = _normalize_plane(a, b, c, d)
        # If opposite to stored normal direction, flip both (n, d)
        if np.dot(n_hat, n_unit) < 0:
            n_hat = -n_hat
            d_hat = -d_hat
        planes.append({
            "n": n_hat,        
            "d": d_hat,         
            "idxs": np.asarray(idxs, dtype=int)
        })

    # Iterative greedy merging: merge any pair that meets the conditions until no more merges occur
    changed = True
    while changed:
        changed = False
        N = len(planes)
        if N <= 1:
            break

        merged_pair = None
        for i in range(N):
            for j in range(i+1, N):
                n1, d1 = planes[i]["n"], planes[i]["d"]
                n2, d2 = planes[j]["n"], planes[j]["d"]

                # To compare angles and offsets, first ensure that n2 and n1 are in the same direction.
                if np.dot(n1, n2) < 0:
                    n2_cmp = -n2
                    d2_cmp = -d2
                else:
                    n2_cmp = n2
                    d2_cmp = d2

                angle = _angle_deg(n1, n2_cmp)
                offset_diff = abs(d1 - d2_cmp)

                if angle <= angle_thresh_deg and offset_diff <= offset_thresh:
                    merged_pair = (i, j)
                    break
            if merged_pair is not None:
                break

        if merged_pair is not None:
            i, j = merged_pair
            # Merge indices of two planes and remove duplicates
            idxs_merged = np.unique(np.concatenate([planes[i]["idxs"], planes[j]["idxs"]], axis=0))

            # Refit the plane using the merged set of points
            pts = all_points_xyz[idxs_merged]
            model_new, n_new = _refit_plane_from_points(pts)
            # Normalize again (refitted normal is already unit length, but renormalizing for stability)
            n_hat, d_hat = _normalize_plane(*model_new)

            # Keep direction consistent with previous normal (using plane i as reference)
            if np.dot(n_hat, planes[i]["n"]) < 0:
                n_hat = -n_hat
                d_hat = -d_hat
                model_new = np.array([n_hat[0], n_hat[1], n_hat[2], d_hat])

            # Construct merged plane, replace plane i, and delete plane j
            planes[i] = {"n": n_hat, "d": d_hat, "idxs": idxs_merged}
            del planes[j]
            changed = True


    new_plane_models = [np.array([pl["n"][0], pl["n"][1], pl["n"][2], pl["d"]], dtype=float) for pl in planes]
    new_normals = [pl["n"] for pl in planes]
    new_indices_list = [pl["idxs"] for pl in planes]
    return new_plane_models, new_normals, new_indices_list

all_points_xyz = np.asarray(pcd.points)  # Original point cloud
plane_models, plane_normals, plane_indices_list = merge_coplanar_planes(
    plane_models,
    plane_normals,
    plane_indices_list,
    all_points_xyz,
    angle_thresh_deg=plane_angle_thresh,
    offset_thresh=offset_thresh
)
print(f"Number of planes after merging:{len(plane_models)}")
for k, (m, n, idxs) in enumerate(zip(plane_models, plane_normals, plane_indices_list)):
    print(f"[{k}] model(a,b,c,d)= {m}, normal= {n}, num_points= {len(idxs)}")

# ---------------- Step 4: Visualization coloring ----------------
# Create a new point cloud and apply colors to all points
colored_pcd = copy.deepcopy(pcd)
colors = np.ones((len(original_points), 3)) * [0.5, 0.5, 0.5]  # Default gray color

for indices, color in zip(plane_indices_list, plane_colors):
    colors[indices] = color  # Assign the plane color to all points belonging to this plane

colored_pcd.colors = o3d.utility.Vector3dVector(colors)

# Create visual arrows for normals
def create_normal_arrow(origin, normal, length=0.02, color=[1, 0, 0]):
    """
    Create a normal arrow for visualization
    origin: starting point (3D coordinates)
    normal: unit normal vector
    length: arrow length
    color: RGB color
    """
    arrow = o3d.geometry.TriangleMesh.create_arrow(
        cylinder_radius=0.001,
        cone_radius=0.002,
        cylinder_height=length * 0.8,
        cone_height=length * 0.2
    )
    arrow.paint_uniform_color(color)

    # Construct rotation matrix that rotates the arrow from the +Z axis to the given normal
    z_axis = np.array([0, 0, 1])
    normal = normal / np.linalg.norm(normal)
    v = np.cross(z_axis, normal)
    c = np.dot(z_axis, normal)
    if np.linalg.norm(v) < 1e-6:
        R = np.eye(3)
    else:
        vx = np.array([[0, -v[2], v[1]],
                    [v[2], 0, -v[0]],
                    [-v[1], v[0], 0]])
        R = np.eye(3) + vx + vx @ vx * ((1 - c) / (np.linalg.norm(v)**2))

    arrow.rotate(R, center=(0, 0, 0))
    arrow.translate(origin)
    return arrow


arrow_list = []

for indices, normal in zip(plane_indices_list, plane_normals):
    pts = np.asarray(pcd.select_by_index(indices.tolist()).points)
    center = np.mean(pts, axis=0)

    arrow = create_normal_arrow(center, normal, length=0.015, color=[1, 0, 0])
    arrow_list.append(arrow)




# ---------------- (Optional) Output plane index list ----------------
for i, indices in enumerate(plane_indices_list):
    print(f"Plane {i}: {len(indices)} points, indices example: {indices[:5]}")
    # o3d.visualization.draw_geometries([pcd.select_by_index(indices.tolist())], window_name=f"Plane {i} points", width=800, height=600)


# Display planes and normals
if no_image == False:
    if show_all_planes_and_normals == True:
        o3d.visualization.draw_geometries([colored_pcd] + arrow_list, window_name="Plane segmentation result and external normal",point_show_normal=False)

####################################################################


def is_parallel(v1, v2, angle_thresh_deg=5):
    cos_theta = np.clip(np.dot(v1, v2), -1.0, 1.0)
    angle = degrees(acos(abs(cos_theta)))  # abs ensures that vectors with opposite directions (±) are also treated as parallel.
    return angle <= angle_thresh_deg

unclustered = set(range(len(plane_normals)))
parallel_groups = []

while unclustered:
    idx = unclustered.pop()
    ref_normal = plane_normals[idx]

    current_group = [idx]

    to_remove = []
    for other in unclustered:
        if is_parallel(ref_normal, plane_normals[other],2*plane_angle_thresh):
            current_group.append(other)
            to_remove.append(other)

    for i in to_remove:
        unclustered.remove(i)

    parallel_groups.append(current_group)



colored_pcd = copy.deepcopy(pcd)
colors = np.ones((len(pcd.points), 3)) * [0.5, 0.5, 0.5]  # Default gray background.

group_colors = [[random.random(), random.random(), random.random()] for _ in parallel_groups]

for group_idx, group in enumerate(parallel_groups):
    color = group_colors[group_idx]
    for plane_idx in group:
        point_indices = plane_indices_list[plane_idx]
        colors[point_indices] = color

colored_pcd.colors = o3d.utility.Vector3dVector(colors)

if no_image == False:
    if show_planes_parallel_clustering == True:
        o3d.visualization.draw_geometries([colored_pcd], window_name="Planeclustering result and external normal")

###############################################Plane pairing


def is_opposite_direction(idx_i, idx_j):

    pn1 = np.asarray(plane_normals[idx_i])
    pn2 = np.asarray(plane_normals[idx_j])

    point_indices_i = plane_indices_list[idx_i]
    point_indices_j = plane_indices_list[idx_j]
    plane_points_i = np.asarray(pcd.select_by_index(point_indices_i).points)
    plane_points_j = np.asarray(pcd.select_by_index(point_indices_j).points)
    pc1 = np.mean(plane_points_i, axis=0)
    pc2 = np.mean(plane_points_j, axis=0)

    c1c2 = pc2 - pc1 # Center-to-center vector from plane 1 to plane 2

    dot0 = np.dot(pn1, pn2)
    dot1 = np.dot(c1c2, pn1) # Should be negative when using outward normals.
    dot2 = np.dot(c1c2, pn2) # Should be positive when using outward normals.


    if dot0 < 0 and dot1 < 0 and dot2 > 0:  # Directions are consistent.
        return True
    else:   
        return False



paired_planes = []  # All plane pair

for group in parallel_groups:
    n = len(group)
    for i in range(n):
        for j in range(i + 1, n):  # Prevent repetition of (i,j) and (j,i)
            idx_i = group[i]
            idx_j = group[j]

            n1 = plane_normals[idx_i]
            n2 = plane_normals[idx_j]

            # if is_opposite_direction(idx_i, idx_j):
            paired_planes.append((idx_i, idx_j))

print(f"\n\n=======================================\n========== Paired planes: {len(paired_planes)} ==========\n=======================================")

# Visualization.
for count, (i, j) in enumerate(paired_planes):
    # Create a new color array initialized to gray.
    colors = np.ones((len(pcd.points), 3)) * [0.6, 0.6, 0.6]

    # Assign a color to the current pair.
    color = [random.random(), random.random(), random.random()]
    for idx in plane_indices_list[i]:
        colors[idx] = color
    for idx in plane_indices_list[j]:
        colors[idx] = color

    # Create a new point cloud and assign colors.
    paired_pcd = copy.deepcopy(pcd)
    paired_pcd.colors = o3d.utility.Vector3dVector(colors)

    print(f"Show pair：{i} ↔ {j}")

    if no_image == False:
        if show_plane_pairs == True:
            o3d.visualization.draw_geometries([paired_pcd], window_name=f"Pair {count+1}: Plane {i} and Plane {j}",width=800,height=600,)

#---------------Find center plane---------------

store = GraspStore()
candidate_TCP_all_pairs = []
# iii=11
counter = 0
for iii in range(len(paired_planes)):
# for iii in range(2,3):
    counter+=1  
    print(f"\n\n----------------------------------------\n-------- Processing pair: {counter}/{len(paired_planes)} --------\n----------------------------------------")

    (mmm,nnn) = paired_planes[iii]
    plane_i_points = np.asarray(pcd.select_by_index(plane_indices_list[mmm]).points)
    plane_j_points = np.asarray(pcd.select_by_index(plane_indices_list[nnn]).points)
    center_i = np.mean(plane_i_points, axis=0)
    center_j = np.mean(plane_j_points, axis=0)

    dist_plane = abs(np.dot(center_i-center_j,plane_normals[mmm]))

    print(f"Plane pair distance: {dist_plane:.3f},open:{(f_pg - 2 * w_pg)},close:{g_pg}")
    obb = pcd.get_oriented_bounding_box()
    obb_extent = max(obb.extent)
    print(f"OBB extent:{obb_extent}")

    if dist_plane < 3.0:
        print("\n####################\nPlane pair distance is too close ( < 3.0 mm ), skip\n##########\n")
        continue
    elif dist_plane < g_pg:
        print("\n####################\nPlane pair distance is too small to grasp, skip\n####################\n")
        continue
    elif dist_plane > (f_pg - 2 * w_pg):
        print("\n####################\nPlane pair distance is too far, skip\n####################\n")
        continue

    center_ij = (center_i + center_j) / 2

    dist_dir_i = np.dot(center_ij - center_i,plane_normals[mmm])
    dist_dir_i = -1.0 if dist_dir_i > 0 else 1.0
    dist_dir_j = np.dot(center_ij - center_j,plane_normals[nnn])
    dist_dir_j = -1.0 if dist_dir_i > 0 else 1.0

    dist_i = abs(np.dot((center_ij - center_i),plane_normals[mmm]))
    dist_j = abs(np.dot((center_ij - center_j),plane_normals[nnn]))

    # project_i_dir = (center_ij - center_i) / np.linalg.norm(center_ij - center_i)
    # project_j_dir = (center_ij - center_j) / np.linalg.norm(center_ij - center_j)

    projected_i_points = plane_i_points - dist_dir_i*np.outer(dist_i,plane_normals[mmm])

    projected_j_points = plane_j_points - dist_dir_j*np.outer(dist_j,plane_normals[nnn])

    pcd_proj_i = o3d.geometry.PointCloud()
    pcd_proj_i.points = o3d.utility.Vector3dVector(projected_i_points)
    pcd_proj_i.paint_uniform_color([0, 1, 0])  # green

    pcd_proj_j = o3d.geometry.PointCloud()
    pcd_proj_j.points = o3d.utility.Vector3dVector(projected_j_points)
    pcd_proj_j.paint_uniform_color([1, 0, 0])  # red

    pcd_orig_i = pcd.select_by_index(plane_indices_list[mmm])
    pcd_orig_j = pcd.select_by_index(plane_indices_list[nnn])

    pcd_orig_i.paint_uniform_color([0.85, 0.85, 0.85])  # gray
    pcd_orig_j.paint_uniform_color([0.85, 0.85, 0.85])

    if no_image == False:
        if show_plane_pair_and_proj_in_pcd == True:
            o3d.visualization.draw_geometries([
                pcd.translate([0,0.001,0]),
                pcd_orig_i,
                pcd_orig_j,
                pcd_proj_i,
                pcd_proj_j
            ], window_name="Plane Projection", width=800, height=600)

    if no_skip == False:
        if input("Skip this pair? (y/n)") == "y":
            continue

    #**************************** Plane 1: Project planes and find overlap region ****************************

    def extract_overlap_region(proj_A, proj_B, threshold=0.001,remove = False):
        """
        Extract the overlapping region from two projected point clouds and return the merged overlap point cloud.
        """

        dA = np.asarray(proj_A.compute_nearest_neighbor_distance())
        dB = np.asarray(proj_B.compute_nearest_neighbor_distance())
        print("median spacing A/B:", np.median(dA), np.median(dB))

        threshold = 1.2 * max(np.median(dA), np.median(dB))
        # Build a KDTree.
        kdtree_B = o3d.geometry.KDTreeFlann(proj_B)
        kdtree_A = o3d.geometry.KDTreeFlann(proj_A)

        points_A = np.asarray(proj_A.points)
        points_B = np.asarray(proj_B.points)

        # Points in A that have nearby points in B.
        matched_A = []
        dismatched_A = []
        for p in points_A:
            [_, idx, _] = kdtree_B.search_radius_vector_3d(p, threshold)
            if len(idx) > 0:
                matched_A.append(p)
            else:
                dismatched_A.append(p)

        if remove == True:
            pcd_remove_overlap = o3d.geometry.PointCloud()
            pcd_remove_overlap.points = o3d.utility.Vector3dVector(dismatched_A)
            pcd_remove_overlap.paint_uniform_color([1, 0, 0])

            return pcd_remove_overlap
        else:
            # Points in B that have nearby points in A.
            matched_B = []
            for p in points_B:
                [_, idx, _] = kdtree_A.search_radius_vector_3d(p, threshold)
                if len(idx) > 0:
                    matched_B.append(p)

            A_keep = np.array(matched_A, dtype=float).reshape(-1, 3)
            B_keep = np.array(matched_B, dtype=float).reshape(-1, 3)
            if A_keep.size == 0 and B_keep.size == 0:
                print("\n############################\nThere is no intersection between this pair of planes.\n############################\n")
                return None
            else:
                # Merge points in overlapping regions
                overlap_points = A_keep if B_keep.size == 0 else (B_keep if A_keep.size == 0 else np.vstack([A_keep, B_keep]))

            if len(overlap_points) < 50:
                print("\n############################\nThere are less than 50 points here.\n############################\n")
                return None
            
            pcd_overlap = o3d.geometry.PointCloud()
            pcd_overlap.points = o3d.utility.Vector3dVector(overlap_points)
            pcd_overlap.paint_uniform_color([0, 1, 0])  # Marked in green.

            return pcd_overlap

    overlap_pcd_unfilter = extract_overlap_region(pcd_proj_i, pcd_proj_j, threshold=0.001)
    if overlap_pcd_unfilter is None:
        continue

    overlap_pcd,ind_p1 = filter_by_normal_orientation(overlap_pcd_unfilter,plane_normals[mmm])
    overlap_pcd,ind_p1 = remove_pcd_outlier_statistical(overlap_pcd_unfilter)
    projected_points_p1 = np.asarray(overlap_pcd.points)
    o3d.io.write_point_cloud("overlap_pcd.pcd", overlap_pcd)

    colors = np.ones((len(overlap_pcd_unfilter.points), 3)) * [1,1,0]
    colors[ind_p1,:] = [0,1,0]
    overlap_pcd_unfilter.colors = o3d.utility.Vector3dVector(colors)

    if no_image == False:
        if show_proj_pts_p1 == True:
            # o3d.visualization.draw_geometries([overlap_pcd_unfilter.translate([0,0,0.00001]),pcd_orig_i,pcd_orig_j],window_name="Pair of Planes and Their Overlap Region")            
            o3d.visualization.draw_geometries([overlap_pcd_unfilter.translate([0,0,0.00001])],window_name="Pair of Planes and Their Overlap Region")


    #**************************** Plane 2: Find points between planes ****************************
    def project_points_to_plane(points, plane_point, plane_normal):
        v = points - plane_point
        d = np.dot(v, plane_normal)
        return points - np.outer(d, plane_normal)

    def select_points_between_planes(pcd, center_i, center_j, plane_normal, margin=0.0015, include_planes=True):
        """
        Select points from the full point cloud that lie between two planes.
        """

        if isinstance(pcd, o3d.geometry.PointCloud):
            points = np.asarray(pcd.points)
        elif isinstance(pcd, np.ndarray):
            points = pcd
        # center_i = plane_i_pts.mean(axis=0)
        # center_j = plane_j_pts.mean(axis=0)

        # Distance vector between the planes (direction must be consistent with the normal).
        dist_vec = center_j - center_i
        dist_vec /= np.linalg.norm(dist_vec)
        
        # Project each point onto the normal to obtain its distances to the two planes.
        d_i = np.dot(points - center_i, plane_normal)
        d_j = np.dot(points - center_j, plane_normal)

        # Determine whether each point lies between the two planes (with a margin tolerance).
        if include_planes:
            mask = (d_i * d_j <= 0) | (np.abs(d_i) <= margin) | (np.abs(d_j) <= margin)
        else:
            mask = (d_i * d_j < 0) & (np.abs(d_i) > margin) & (np.abs(d_j) > margin)

        points_between = points[mask]
        points_beside = points[~mask]
        return points_between,points_beside




    # 1. Select points that lie between the two planes.
    points_between_p2,points_beside = select_points_between_planes(pcd, center_i, center_j, plane_normals[mmm],margin=margin_points_between_planes)

    # 2. Project onto the middle plane.
    projected_points_p2 = project_points_to_plane(points_between_p2, center_ij, plane_normals[mmm])

    # 3. Create a PointCloud object.
    proj_pcd_p2_unfilter = o3d.geometry.PointCloud()
    proj_pcd_p2_unfilter.points = o3d.utility.Vector3dVector(projected_points_p2)
    proj_pcd_p2_unfilter.paint_uniform_color([1, 0, 0])  

    proj_pcd_p2,ind_p2 = remove_pcd_outlier_dbscan(proj_pcd_p2_unfilter)
    projected_points_p2 = np.asarray(proj_pcd_p2.points)

    colors = np.ones((len(proj_pcd_p2_unfilter.points), 3)) * [1,1,0]
    colors[ind_p2,:] = [1,0,0] # red
    proj_pcd_p2_unfilter.colors = o3d.utility.Vector3dVector(colors)

    # visualization
    if no_image == False:
        if show_proj_pts_p2 == True:
            # o3d.visualization.draw_geometries([
            #     pcd_orig_j, pcd_orig_i, proj_pcd_p2_unfilter
            #     ,
            # ],window_name="Pair of Planes and Projected Points Between Them")            
            o3d.visualization.draw_geometries([
                proj_pcd_p2_unfilter
                ,
            ],window_name="Pair of Planes and Projected Points Between Them")


    #**************************** Plane 3: find within finger width collision area ****************************


    center_i_p3 = center_i + (a_pg + w_pg + v_pg - roughness_tolerance) * (plane_normals[mmm]) * dist_dir_i
    center_j_p3 = center_j + (a_pg + w_pg + v_pg - roughness_tolerance) * (plane_normals[nnn]) * dist_dir_j
    # center_i_p3 = center_ij + (0.02) * (plane_normals[mmm]) * dist_dir_i
    # center_j_p3 = center_ij + (0.02) * (plane_normals[nnn]) * dist_dir_j

    # 1. Select points that lie between the two planes.
    points_between_p3_i,points_beside = select_points_between_planes(points_beside, center_i, center_i_p3, plane_normals[mmm],margin=margin_points_between_planes)
    points_between_p3_j,points_beside = select_points_between_planes(points_beside, center_j, center_j_p3, plane_normals[nnn],margin=margin_points_between_planes)
    points_between_p3 = np.vstack((points_between_p3_i, points_between_p3_j))

    # 2. Project onto the middle plane.
    projected_points_p3 = project_points_to_plane(points_between_p3, center_ij, plane_normals[mmm])

    # 3. Create a PointCloud object.
    proj_pcd_p3_unfilter = o3d.geometry.PointCloud()
    proj_pcd_p3_unfilter.points = o3d.utility.Vector3dVector(projected_points_p3)
    # proj_pcd_p3_unfilter.paint_uniform_color([0, 0, 1])

    proj_pcd_p3,ind_p3 = remove_pcd_outlier_dbscan(proj_pcd_p3_unfilter)
    projected_points_p3 = np.asarray(proj_pcd_p3.points)

    o3d.io.write_point_cloud("proj_pcd_p3.pcd", proj_pcd_p3)

    colors = np.ones((len(proj_pcd_p3_unfilter.points), 3)) * [1,1,0]
    colors[ind_p3,:] = [0,0,1] # blue
    proj_pcd_p3_unfilter.colors = o3d.utility.Vector3dVector(colors)


    if no_image == False:
        if show_proj_pts_p3 == True:
            # o3d.visualization.draw_geometries([overlap_pcd, proj_pcd_p3_unfilter.translate([0,0,0.0001])],window_name="Initial TCP & Finger Collision Area (P3)")            
            o3d.visualization.draw_geometries([ proj_pcd_p3_unfilter.translate([0,0,0.0001])],window_name="Initial TCP & Finger Collision Area (P3)")

    #**************************** Plane 4: find outside finger width collision area ****************************

    center_i_p4 = center_ij + (y_pg/2) * (plane_normals[mmm]) * dist_dir_i
    center_j_p4 = center_ij + (y_pg/2) * (plane_normals[nnn]) * dist_dir_j
    # center_i_p3 = center_ij + (0.02) * (plane_normals[mmm]) * dist_dir_i
    # center_j_p3 = center_ij + (0.02) * (plane_normals[nnn]) * dist_dir_j

    # 1. Select points that lie between the two planes.
    points_between_p4_i,points_beside = select_points_between_planes(points_beside, center_i_p3, center_i_p4, plane_normals[mmm],margin=margin_points_between_planes)
    points_between_p4_j,points_beside = select_points_between_planes(points_beside, center_j_p3, center_j_p4, plane_normals[nnn],margin=margin_points_between_planes)
    points_between_p4 = np.vstack((points_between_p4_i, points_between_p4_j))

    # 2. Project onto the middle plane.
    projected_points_p4 = project_points_to_plane(points_between_p4, center_ij, plane_normals[mmm])

    # 3. Create a PointCloud object.
    proj_pcd_p4_unfilter = o3d.geometry.PointCloud()
    proj_pcd_p4_unfilter.points = o3d.utility.Vector3dVector(projected_points_p4)
    # proj_pcd_p4_unfilter.paint_uniform_color([0, 0, 1])  # green

    proj_pcd_p4,ind_p4 = remove_pcd_outlier_dbscan(proj_pcd_p4_unfilter)
    projected_points_p4 = np.asarray(proj_pcd_p4.points)

    o3d.io.write_point_cloud("proj_pcd_p4.pcd", proj_pcd_p4)

    colors = np.ones((len(proj_pcd_p4_unfilter.points), 3)) * [1,1,0]
    colors[ind_p4,:] = [0,0.5,1] # deep sky blue
    proj_pcd_p4_unfilter.colors = o3d.utility.Vector3dVector(colors)

    if no_image == False:
        if show_proj_pts_p4 == True:
            # o3d.visualization.draw_geometries([overlap_pcd,proj_pcd_p3_unfilter.translate([0,0,-0.0001]), proj_pcd_p4_unfilter.translate([0,0,0.0001])],window_name="Initial TCP & Finger Collision Area (P3+P4)")            
            o3d.visualization.draw_geometries([proj_pcd_p4_unfilter.translate([0,0,0.0001])],window_name="Initial TCP & Finger Collision Area (P3+P4)")

    ##**************************** Plane 5: find beside collision area ****************************

    center_i_p5 = center_ij + ((rd + rj)/2) * (plane_normals[mmm]) * dist_dir_i
    center_j_p5 = center_ij + ((rd + rj)/2) * (plane_normals[nnn]) * dist_dir_j

    points_between_p5_i,points_beside = select_points_between_planes(points_beside, center_i_p4, center_i_p5, plane_normals[mmm],margin=margin_points_between_planes)
    points_between_p5_j,points_beside = select_points_between_planes(points_beside, center_j_p4, center_j_p5, plane_normals[nnn],margin=margin_points_between_planes)
    points_between_p5 = np.vstack((points_between_p5_i, points_between_p5_j))
    
    projected_points_p5 = project_points_to_plane(points_between_p5, center_ij, plane_normals[mmm])


    # 3. Create a PointCloud object.
    proj_pcd_p5_unfilter = o3d.geometry.PointCloud()
    proj_pcd_p5_unfilter.points = o3d.utility.Vector3dVector(projected_points_p5)
    proj_pcd_p5_unfilter.paint_uniform_color([0, 1, 1])

    proj_pcd_p5,ind_p5 = remove_pcd_outlier_dbscan(proj_pcd_p5_unfilter)
    # proj_pcd_p5.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
    o3d.io.write_point_cloud("proj_pcd_p5.pcd", proj_pcd_p5)
    projecter_points_p5 = np.asarray(proj_pcd_p5.points)

    colors = np.ones((len(proj_pcd_p5_unfilter.points),3)) * [1,1,0]
    colors[ind_p5,:] = [0,1,1] # cyan
    proj_pcd_p5_unfilter.colors = o3d.utility.Vector3dVector(colors)

    if no_image == False:
        if show_proj_pts_p5 == True:
            o3d.visualization.draw_geometries([overlap_pcd, proj_pcd_p3.translate([0,0,0.0001]), proj_pcd_p4.translate([0,0,0.0002]), proj_pcd_p5_unfilter.translate([0,0,-0.0001])],window_name="Initial TCP & Finger Collision Area (P3+P4) & Robot Collision Area (P5)")

    #********************** SHOW P1 P2 P3 P4 P5 WITH PCD ****************

    pcd_between_p22 = o3d.geometry.PointCloud()
    pcd_between_p22.points = o3d.utility.Vector3dVector(points_between_p2)
    pcd_between_p22.paint_uniform_color([1, 0, 0])  # red

    pcd_between_p33 = o3d.geometry.PointCloud()
    pcd_between_p33.points = o3d.utility.Vector3dVector(points_between_p3)
    pcd_between_p33.paint_uniform_color([0, 0, 1])  # blue

    pcd_between_p44 = o3d.geometry.PointCloud()
    pcd_between_p44.points = o3d.utility.Vector3dVector(points_between_p4)
    pcd_between_p44.paint_uniform_color([0, 0.5, 1])  # deep sky blue

    pcd_beside_p5 = o3d.geometry.PointCloud()
    pcd_beside_p5.points = o3d.utility.Vector3dVector(points_beside)
    pcd_beside_p5.paint_uniform_color([0, 1, 1])  # cyan

    if no_image == False:
        if show_P2345_in_pcd == True:
            o3d.visualization.draw_geometries([pcd_between_p22, pcd_between_p33.translate([0,0,0.0001]), pcd_between_p44.translate([0,0,0.0002]), pcd_beside_p5.translate([0,0,-0.0001])],window_name="PCD [P2 P3 P4 P5] in 3D")

    #*Show projected points on P1 P2 P3 P4 P5 with assemble PCD in 3D

    pcd.paint_uniform_color([0.85, 0.85, 0.85])

    if no_image == False:
        if show_each_P_in_pcd == True: 
            o3d.visualization.draw_geometries([pcd, overlap_pcd.translate([0,0,0.0001])],window_name="PCD+P1")
            o3d.visualization.draw_geometries([pcd, proj_pcd_p2.translate([0,0,0.0001])],window_name="PCD+P2")
            o3d.visualization.draw_geometries([pcd, proj_pcd_p3.translate([0,0,-0.0001])],window_name="PCD+P3")
            o3d.visualization.draw_geometries([pcd, proj_pcd_p4.translate([0,0,-0.0001])],window_name="PCD+P4")
            o3d.visualization.draw_geometries([pcd, proj_pcd_p5.translate([0,0,-0.0001])],window_name="PCD+P5")

    #**************************** P2: Find contours ****************************

    def auto_img_scale(pcd, target_size=512):
        points = np.asarray(pcd.points)
        # Compute the width and height of the original point cloud in the 2D principal plane.
        min_vals = points.min(axis=0)
        max_vals = points.max(axis=0)
        ranges = max_vals - min_vals

        # Use the longer side as the reference and scale it to target_size.
        scale = target_size / np.max(ranges)
        return 1
    #-----------------------cv2------------------------

    def _detect_plane_2d_contours(pcd, dir1, dir2, center):
        """
        Shared core pipeline: project -> rasterize -> morphology -> connected-component
        filter -> 2-level contour detection (outer + first-level holes).
        Runs ONCE per point cloud and is consumed by both _build_plane_polygons (outer+hole
        polygons) and _build_plane_boundary_segments (outer boundary segments/normals), so
        a plane that needs both views (P2) no longer pays for the image pipeline twice.
        """
        if pcd.is_empty():
            return None

        points = np.asarray(pcd.points)
        if points.size <= 50:
            return None

        # Project onto the principal plane (2D).
        points_2d = np.dot(points - center, np.vstack([dir1, dir2]).T)

        min_pt = points_2d.min(axis=0)
        max_pt = points_2d.max(axis=0)

        # Convert points to image coordinates.
        img_scale = auto_img_scale(pcd)
        points_img = np.int32((points_2d - min_pt) * img_scale) + contour_image_padding
        img_size = ((max_pt - min_pt) * img_scale).astype(int) + 2 * contour_image_padding

        # Create a blank image and plot the points.
        img = np.zeros((img_size[1], img_size[0]), dtype=np.uint8)
        for pt in points_img:
            cv2.circle(img, tuple(pt), 1, 255, -1)

        # ---- Estimate a typical pixel spacing between points (for choosing the kernel size). ----
        px_gap = 3  # Simplified: fall back to a fixed constant (kept identical to original behavior).

        # ---- Morphology: perform closing first and then opening. ----
        k = max(3, int(round(px_gap * 2)))         # core for closing operation (the larger it is, the more it "completes")
        k_open = max(3, int(round(px_gap * 0.8)))  # core for opening operation (light noise reduction)
        kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        kernel_open  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_open, k_open))

        mask = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel_close, iterations=1)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  kernel_open,  iterations=1)

        # ---- Remove small connected components (to prevent isolated points from affecting the outer contour). ----
        num, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        min_area_px = (k * k) * 2  # Area threshold: related to the kernel size.
        clean = np.zeros_like(mask)
        for i in range(1, num):
            if stats[i, cv2.CC_STAT_AREA] >= min_area_px:
                clean[labels == i] = 255

        # ---- Detect contours with a 2-level hierarchy: level 1 = outer boundaries, level 2 = holes. ----
        # The top-level (parent == -1) subset of a RETR_CCOMP result is exactly equivalent to what
        # RETR_EXTERNAL would return, so this single pass also covers what the old P2-only
        # boundary-segment extraction needed RETR_EXTERNAL for.
        contours, hierarchy = cv2.findContours(clean, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
        if hierarchy is None or len(contours) == 0:
            return None
        hierarchy = hierarchy[0]   # shape (N, 4) -> [next, prev, first_child, parent]

        # ---- Pre-compute polygon approximation for every contour (outer + holes) ----
        approx_img_list = []
        for cnt in contours:
            epsilon = 0.01 * cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, epsilon, True).reshape(-1, 2)
            approx_img_list.append(approx)

        def img_to_3d(approx_img):
            """image coords -> 2D projected coords -> original 3D coords"""
            pts_2d = (approx_img.astype(np.float32) - contour_image_padding) / img_scale + min_pt
            pts_3d = np.dot(pts_2d, np.vstack([dir1, dir2])) + center
            return pts_2d, pts_3d

        return {
            "img": img,
            "contours": contours,
            "hierarchy": hierarchy,
            "approx_img_list": approx_img_list,
            "img_to_3d": img_to_3d,
        }

    def _build_plane_polygons(detection, pcd, plane_name=""):
        """
        Build the list of (possibly holed) shapely Polygons from a shared detection result.
        This is the "outer contour, with first-level hole embedded" view used by every plane
        (P1-P5) for the geometric feasibility checks.
        """
        if detection is None:
            return []

        contours = detection["contours"]
        hierarchy = detection["hierarchy"]
        approx_img_list = detection["approx_img_list"]
        img_to_3d = detection["img_to_3d"]
        img = detection["img"]

        def make_lineset(points_3d, color):
            n = points_3d.shape[0]
            lines = [[i, (i + 1) % n] for i in range(n)]
            colors = [color for _ in lines]
            ls = o3d.geometry.LineSet()
            ls.points = o3d.utility.Vector3dVector(points_3d)
            ls.lines = o3d.utility.Vector2iVector(lines)
            ls.colors = o3d.utility.Vector3dVector(colors)
            return ls

        polygons_2d = []   # Each outer contour corresponds to a polygon; if there is an inner contour in the first layer, it is embedded as a hole
        linesets = []
        img_contours = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

        for idx, cnt in enumerate(contours):
            parent = hierarchy[idx][3]
            if parent != -1:
                # This is a hole contour, processed together with its parent (outer) contour; skip here.
                continue

            approx_outer_img = approx_img_list[idx]
            if approx_outer_img.shape[0] < 3:
                continue

            outer_2d, outer_3d = img_to_3d(approx_outer_img)
            cv2.drawContours(img_contours, [approx_outer_img.astype(np.int32)], 0, (0, 255, 255), 2)  # Blue: Outer contour
            linesets.append(make_lineset(outer_3d, [0, 1, 1]))  # Blue: Outer contour 3D wireframe

            # --- Select only the first hole (first_child) ---
            holes_2d = []
            child_idx = hierarchy[idx][2]
            while child_idx != -1:
                approx_inner_img = approx_img_list[child_idx]
                if approx_inner_img.shape[0] >= 3:
                    inner_2d, inner_3d = img_to_3d(approx_inner_img)
                    holes_2d.append(inner_2d)
                    cv2.drawContours(img_contours, [approx_inner_img.astype(np.int32)], 0, (255, 0, 0), 2)  # Red: Inner contour (2D)
                    linesets.append(make_lineset(inner_3d, [1, 0, 0]))  # Red: Inner contour 3D wireframe
                    child_idx = hierarchy[child_idx][0]

            # --- Combine into a polygon with holes ---
            poly = Polygon(outer_2d, holes=holes_2d) if holes_2d else Polygon(outer_2d)
            if not poly.is_valid:
                poly = poly.buffer(0)

            # buffer(0) can split a degenerate polygon into a MultiPolygon;
            # flatten it so polygons_2d always contains plain Polygon objects.
            if isinstance(poly, MultiPolygon):
                polygons_2d.extend(list(poly.geoms))
            else:
                polygons_2d.append(poly)

        # Display contours and processed results.
        plt.figure(figsize=(6, 6))  # Contours
        plt.imshow(img_contours)
        plt.title('Contour Detection Result: ' + plane_name)
        plt.axis('off')
        if no_image == False and show_plt_contours_Px_2d == True:
            plt.show()
        else:
            plt.close()

        pcd.paint_uniform_color([0.6, 0.6, 0.6])

        if no_image == False:
            if show_P_contour_3d == True:
                o3d.visualization.draw_geometries([pcd, *linesets], window_name='Plane Contours 3D View')

        return polygons_2d

    def _build_plane_boundary_segments(detection, pcd, dir1, dir2, center, plane_name=""):
        """
        Build per-edge 2D line segments + normals (and their 3D counterparts) from a shared
        detection result. This is the P2-only view used for grasp-pose / TCP-grid computation,
        equivalent to what the old extract_and_visualize_contour_segments_with_normals produced,
        but it reuses the detection result instead of re-running the image pipeline.
        """
        contours = detection["contours"]
        hierarchy = detection["hierarchy"]
        approx_img_list = detection["approx_img_list"]
        img_to_3d = detection["img_to_3d"]
        img = detection["img"]

        # The top-level (parent == -1) contours here are exactly what RETR_EXTERNAL used to return.
        outer_idxs = [idx for idx in range(len(contours)) if hierarchy[idx][3] == -1]

        # Draw the contour processing results (outer boundary only, matching the old P2-specific debug view).
        img_contours = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        for idx in outer_idxs:
            approx = approx_img_list[idx]
            cv2.drawContours(img_contours, [approx.astype(np.int32)], 0, (255, 0, 0), 2)

        plt.figure(figsize=(6, 6))  # Contour P2
        plt.imshow(img_contours)
        plt.title('Contours: ' + plane_name)
        plt.axis('off')
        if no_image == False and show_plt_contour_P2_2d == True:
            plt.show()
        else:
            plt.close()

        line_segments_2d, line_normals_2d = [], []
        line_segments_3d, line_indices, line_colors = [], [], []
        line_set = o3d.geometry.LineSet()

        # NOTE: kept identical to the original behavior - if there are multiple disconnected
        # outer contours, only the LAST one's segments are kept (the original loop overwrote
        # line_segments_2d/line_normals_2d on every iteration). Flag this if you ever want all
        # outer contours' edges instead of just the last one.
        for idx in outer_idxs:
            approx_outer_img = approx_img_list[idx]
            points_2d_back, _ = img_to_3d(approx_outer_img)

            line_segments_2d, line_normals_2d = [], []
            line_segments_3d, line_indices, line_colors = [], [], []

            n_pts = len(points_2d_back)
            for i in range(n_pts):
                pt1_2d = points_2d_back[i]
                pt2_2d = points_2d_back[(i + 1) % n_pts]

                # Line segment direction and its normal.
                vec = pt2_2d - pt1_2d
                length = np.linalg.norm(vec)
                if length == 0:
                    continue
                direction = vec / length
                normal_2d = np.array([-direction[1], direction[0]])

                line_segments_2d.append([pt1_2d, pt2_2d])
                line_normals_2d.append(normal_2d)

                # Project back into 3D space.
                pt1_3d = center + pt1_2d[0]*dir1 + pt1_2d[1]*dir2
                pt2_3d = center + pt2_2d[0]*dir1 + pt2_2d[1]*dir2

                # Line segment added to LineSet
                idx_pt = len(line_segments_3d)
                line_segments_3d.extend([pt1_3d, pt2_3d])
                line_indices.append([idx_pt, idx_pt + 1])
                color = plt.cm.hsv(i / n_pts)[:3]
                line_colors.append(color)

            if len(line_indices) == 0:
                print("No segments to visualize (the contour might be too small or overly simplified).")
                continue

            line_set = o3d.geometry.LineSet()
            line_set.points = o3d.utility.Vector3dVector(np.asarray(line_segments_3d, dtype=float))
            line_set.lines = o3d.utility.Vector2iVector(np.asarray(line_indices, dtype=np.int32))
            line_set.colors = o3d.utility.Vector3dVector(np.asarray(line_colors, dtype=float))

        # Visualization
        if no_image == False:
            if show_P2_contour_3d == True:
                o3d.visualization.draw_geometries([pcd, line_set], window_name="P2 Contour Lines + Normals", width=1280, height=800)

        return line_segments_2d, line_normals_2d, [line_segments_3d, line_indices]

    def extract_and_visualize_contour_segments_with_normals(pcd, scale=1500, approx_eps_ratio=0.01):
        """
        P2-only entry point: computes dir1/dir2/center via PCA (the single source of truth
        reused by every other plane), then runs the shared contour-detection pipeline ONCE and
        derives BOTH the boundary-segments view (for grasp-pose/TCP-grid computation) AND the
        polygon-with-hole view (for the geometric feasibility checks) from that single pass,
        instead of detecting P2's contours twice (once here, once again later in
        get_plane_contour_polygon).
        """
        if isinstance(pcd, o3d.geometry.PointCloud):
            points = np.asarray(pcd.points)
        elif isinstance(pcd, np.ndarray):
            print("Find contours Error: Input is not a PointCloud object.")
            return

        # 1. PCA principal directions (dir1, dir2 form the local plane).
        pca = PCA(n_components=3)
        pca.fit(points)
        dir1, dir2 = pca.components_[0], pca.components_[1]
        center = pca.mean_

        # 2. Run the shared detection pipeline exactly once.
        detection = _detect_plane_2d_contours(pcd, dir1, dir2, center)

        # 3a. Boundary segments + normals (P2-specific downstream use).
        line_segments_2d, line_normals_2d, segments_3d_para = _build_plane_boundary_segments(
            detection, pcd, dir1, dir2, center, plane_name="Plane 2"
        )

        # 3b. Polygon-with-hole list (same representation used for P1/P3/P4/P5).
        polygons_2d_p2 = _build_plane_polygons(detection, pcd, plane_name="Plane 2")

        return line_segments_2d, line_normals_2d, dir1, dir2, center, segments_3d_para, polygons_2d_p2


    contour_segments_2d_p2 = []
    contour_normals_2d_p2 = []
    contour_segments_2d_p2,contour_normals_2d_p2,dir1,dir2,center, contour_segments_3d_p2_para, plane2_polygons = extract_and_visualize_contour_segments_with_normals(proj_pcd_p2, scale=1500, approx_eps_ratio=0.01)


    #***************************** Compute grasp pose of each contour segment *********************************

    def safe_normalize(v, eps=1e-9):
        v = np.asarray(v, dtype=float)
        n = np.linalg.norm(v)
        if n < eps:
            return None
        return v / n

    def compute_grasp_pose(line_segments_2d, line_normals_2d, dir1, dir2, center):
        """
        Mapping the 2D segment back to 3D
        Calculate the grasp pose coordinate for each segment

        Parameters
        ----------
        line_segments_2d : list of [[pt1_2d, pt2_2d], ...]

        line_normals_2d : list of [nx, ny]

        dir1, dir2 : np.ndarray shape (3,)
            PCA principal plane direction vector (projection plane base)
        center : np.ndarray shape (3,)
            Projection center (only used for mapping points back to 3D)

        Returns
        -------
        grasp_frame : list of [seg3d, normal3d, plane_normal3d]
            
        """

        dir1 = np.asarray(dir1, dtype=float)
        dir2 = np.asarray(dir2, dtype=float)
        center = np.asarray(center, dtype=float)

        segs = list(line_segments_2d)
        norms2d = list(line_normals_2d)
        N = min(len(segs), len(norms2d))

        grasp_frame = []

        for i in range(N):
            pt1_2d = np.asarray(segs[i][0], dtype=float)
            pt2_2d = np.asarray(segs[i][1], dtype=float)
            n2d = np.asarray(norms2d[i], dtype=float)

            # ---- Back projection back to 3D ----
            pt1_3d = center + pt1_2d[0] * dir1 + pt1_2d[1] * dir2
            pt2_3d = center + pt2_2d[0] * dir1 + pt2_2d[1] * dir2


            v3d = pt2_3d - pt1_3d
            seg3d = safe_normalize(v3d) # 3D vector of segments
            if seg3d is None:
                continue

            # normal3d = Segment 3d normal direction
            normal3d_raw = n2d[0] * dir1 + n2d[1] * dir2
            normal3d = safe_normalize(-normal3d_raw) # 3D vector of segment's normal #! 2d normal vector is anti clockwise 90 degree with the segment direction, so we need to reverse it.
            if normal3d is None:
                # If it degenerates, then take the plane vector perpendicular to seg3d.
                plane_normal = np.cross(dir1, dir2)
                plane_normal = safe_normalize(plane_normal)
                if plane_normal is not None:
                    normal3d = safe_normalize(np.cross(plane_normal, seg3d))
            if normal3d is None:
                continue

            # plane_normal3d =  normal3d × seg3d
            plane_normal3d = np.cross(normal3d, seg3d)
            plane_normal3d = safe_normalize(plane_normal3d)
            if plane_normal3d is None:
                continue

            # # Reorthogonalization
            # normal3d = np.cross(seg3d, plane_normal3d)
            # normal3d = safe_normalize(normal3d)

            grasp_frame.append([normal3d, seg3d, plane_normal3d]) # corresponding to gripper frame X,Y,Z
            print("Grasp frame:", grasp_frame[-1])
        return grasp_frame
    

    grasp_pose = compute_grasp_pose(contour_segments_2d_p2, contour_normals_2d_p2, dir1, dir2, center)



    # **************************** Find and Show Initial TCP Box & Test Grid Point ****************************
    def generate_grid_by_spacing(segments_2d, normals_2d, depth=0.05, spacing_edge=0.005,spacing_normal=0.005):
        """
        For each line segment, extend it along its normal direction to form a rectangle, and generate evenly spaced grid points inside it with the given spacing.
        
        Parameters:
            segments_2d: List of (pt1, pt2), start and end points of each line segment.
            normals_2d: List of unit normal vectors, one per line segment.
            depth: width of the grasping region along the normal direction (in meters).
            spacing: spacing between grid points (in meters).
            
        Returns:
            rectangles: the rectangle (4 points) corresponding to each line segment.
            all_grid_points: generated grid points inside each rectangle, as a list of np.ndarray.
        """
        rectangles = []
        all_grid_points = []

        eps=1e-9

        for (pt1, pt2), n in zip(segments_2d, normals_2d):
            pt1 = np.array(pt1)
            pt2 = np.array(pt2)
            n = np.array(n) / np.linalg.norm(n)

            # Line segment direction and length.
            dir_vec = pt2 - pt1
            seg_len = np.linalg.norm(dir_vec)
            dir_unit = dir_vec / seg_len

            # Determine the number of steps along the segment direction.
            num_w = int(np.floor((seg_len-eps) / spacing_edge)+1)
            start_spacing_edge = (seg_len-(num_w-1)*spacing_edge)/2.0
            num_d = int(np.floor((depth-eps) / spacing_normal)+1)
            start_spacing_normal = (depth-(num_d-1)*spacing_normal)/2.0      
            if num_w < 1 or num_d < 1:
                continue

            # Construct the four rectangle vertices (counterclockwise).
            offset = -n * depth
            p1 = pt1 + offset
            p2 = pt2 + offset
            p3 = pt2
            p4 = pt1
            rectangles.append([p1, p2, p3, p4])

            # Generate regular grid points inside the rectangle.
            grid_pts = []
            for i in range(num_w):
                for j in range(num_d):
                    alpha = i * spacing_edge + start_spacing_edge
                    beta = j * spacing_normal + start_spacing_normal
                    # alpha = i * spacing_edge
                    # beta = j * spacing_normal
                    pt = p1 + dir_unit * alpha + n * beta
                    grid_pts.append(pt)
            all_grid_points.append(np.array(grid_pts))

        return rectangles, all_grid_points

    def _get_inner_contour_rings(polygons):
        """
        Extract inner-contour (hole) ring coordinate arrays from a list of shapely Polygons.
        Returns a list of (N+1, 2) numpy arrays, one per hole ring found across all polygons.
        Used by the four visualisation functions to draw inner contours in red.
        """

        rings = []
        for poly in (polygons or []):
        # buffer(0) / make_valid may convert a single Polygon into a MultiPolygon
        # Flatten into individual Polygons before further processing
            if isinstance(poly, MultiPolygon):
                sub_polys = list(poly.geoms)
            else:
                sub_polys = [poly]

            for p in sub_polys:
                for interior in p.interiors:
                    rings.append(np.array(interior.coords))

        return rings

    def _draw_inner_contours(ax, inner_rings, used_labels):
        """Draw all inner-contour rings in red on ax, adding a legend entry only once."""
        for ring in (inner_rings or []):
            lbl = 'Inner Contour of Plane 2'
            if lbl not in used_labels:
                ax.plot(ring[:, 0], ring[:, 1], color='#c71585', linewidth=1.5, label=lbl)
                used_labels.add(lbl)
            else:
                ax.plot(ring[:, 0], ring[:, 1], color='#c71585', linewidth=1.5)

    def plot_segments_tcpbox_and_grids(segments_2d, rectangles, grid_points, inner_contour_rings=None):
        """
        Draw in the 2D coordinate plane:

        Original line segments (blue)
        Rectangle region of each segment (green dashed lines)
        Regular grid points inside each rectangle (red x markers)
        """

        all_pts = np.array(list(itertools.chain.from_iterable(segments_2d)))
        min_xy = all_pts.min(axis=0) - plt_graphic_padding
        max_xy = all_pts.max(axis=0) + plt_graphic_padding

        fig, ax = plt.subplots(figsize=(9, 8))# All TCP 

        used_labels = set()  # Track already-added legend labels.
        for (pt1, pt2), rect, grids in zip(segments_2d, rectangles, grid_points):
            # Original line segments
            lbl = 'Outer Contour of Plane2'
            if lbl not in used_labels:
                ax.plot([pt1[0], pt2[0]], [pt1[1], pt2[1]], 'b-', linewidth=3, label=lbl)
                used_labels.add(lbl)
            else:
                ax.plot([pt1[0], pt2[0]], [pt1[1], pt2[1]], 'b-', linewidth=3)

            # Rectangle region (closed loop)
            rect = np.array(rect + [rect[0]])
            lbl = 'TCP Box'
            if lbl not in used_labels:
                ax.plot(rect[:, 0], rect[:, 1], 'g--', linewidth=0.5, label=lbl)
                used_labels.add(lbl)
            else:
                ax.plot(rect[:, 0], rect[:, 1], 'g--', linewidth=0.5)

            # Grid points
            grids = np.array(grids)
            lbl = 'Test Grid Points'
            if lbl not in used_labels:
                ax.plot(grids[:, 0], grids[:, 1], 'rx', markersize=4, label=lbl)
                used_labels.add(lbl)
            else:
                ax.plot(grids[:, 0], grids[:, 1], 'rx', markersize=4)

        ax.set_aspect('equal')
        ax.set_title("All TCP Boxes and all Test Grid Points")
        ax.grid(True)
        _draw_inner_contours(ax, inner_contour_rings, used_labels)
        ax.legend()
        plt.tight_layout()
        pair_path = imageout_basepath/f"pair {iii+1}"
        pair_path.mkdir(exist_ok=True, parents=True)
        if save_image:
            plt.savefig(pair_path/f"possible_tcp_all.svg", format='svg', bbox_inches='tight')
        if (not no_image and show_plt_all_tcp_grids):
            plt.show()
        else:
            plt.close()



    tcp_box,test_grid_points = generate_grid_by_spacing(contour_segments_2d_p2, contour_normals_2d_p2, depth=b_pg+c_pg, spacing_edge=z_pg*spacing_edge_normal_factor, spacing_normal=b_pg*spacing_edge_normal_factor)

    if save_image == True or (not no_image and show_plt_all_tcp_grids) == True:
        _p2_inner_rings = _get_inner_contour_rings(plane2_polygons)
        plot_segments_tcpbox_and_grids(contour_segments_2d_p2,tcp_box,test_grid_points, inner_contour_rings=_p2_inner_rings)

    # Show each TCP Boxes and it's test grid points
    def highlight_segment_rect_grid(segments_2d, rectangles, grid_points, inner_contour_rings = None):
        """
        Always display all line segments, but only highlight the rectangle and grid points corresponding to the current index.
        """
        all_pts = np.array(list(itertools.chain.from_iterable(segments_2d)))
        min_xy = all_pts.min(axis=0) - plt_graphic_padding
        max_xy = all_pts.max(axis=0) + plt_graphic_padding

        for i in range(len(segments_2d)):
            fig, ax = plt.subplots(figsize=(8, 8)) #each TCP 
            ax.set_title(f"TCP Box and Test Grid Points for Edges {i+1}/{len(segments_2d)}")

            # All line segments: blue.
            lbl = 'Outer Contour of Plane2'
            used_labels = set()
            for j , (pt1, pt2) in enumerate(segments_2d):

                if j == i:
                    mid = (pt1 + pt2) / 2
                    vec_12 = pt2 - pt1
                    vec_12 = vec_12 / np.linalg.norm(vec_12)
                    normal_clockwise_90 = [vec_12[1], -vec_12[0]]
                    normal_clockwise_90 = normal_clockwise_90 / np.linalg.norm(normal_clockwise_90)

                    #parallel symbol
                    # start_point_line = mid - normal_clockwise_90 * 0.026
                    # end_point_line = start_point_line + normal_clockwise_90 * 0.015
                    # end_point_base1 = end_point_line + vec_12 * 0.005
                    # end_point_base2 = end_point_line - vec_12 * 0.005
                    # end_point_finger1 = end_point_base1 + normal_clockwise_90 * 0.008
                    # end_point_finger2 = end_point_base2 + normal_clockwise_90 * 0.008

                    #tilt symbol
                    start_point_line = mid - normal_clockwise_90 * tilt_symbol_start_dist
                    end_point_line = start_point_line + normal_clockwise_90 * tilt_symbol_handle_length
                    end_point_base1 = end_point_line + vec_12 * tilt_symbol_finger_width_half - normal_clockwise_90 * tilt_symbol_finger_width_half
                    end_point_base2 = end_point_line - vec_12 * tilt_symbol_finger_width_half + normal_clockwise_90 * tilt_symbol_finger_width_half
                    end_point_finger1 = end_point_base1 + normal_clockwise_90 * tilt_symbol_finger_end_length
                    end_point_finger2 = end_point_base2 + normal_clockwise_90 * tilt_symbol_finger_end_length

                    ax.plot([start_point_line[0], end_point_line[0]], [start_point_line[1], end_point_line[1]], 'm', linewidth=1.5,label='Gripper Direction')
                    ax.plot([end_point_base1[0], end_point_base2[0]], [end_point_base1[1], end_point_base2[1]], 'm', linewidth=1.5)
                    ax.plot([end_point_base1[0], end_point_finger1[0]], [end_point_base1[1], end_point_finger1[1]], 'm', linewidth=1.5)
                    ax.plot([end_point_base2[0], end_point_finger2[0]], [end_point_base2[1], end_point_finger2[1]], 'm', linewidth=1.5)
                    
                if lbl not in used_labels:
                    ax.plot([pt1[0], pt2[0]], [pt1[1], pt2[1]], 'b-', linewidth=3,label=lbl)
                    used_labels.add(lbl)
                else:
                    ax.plot([pt1[0], pt2[0]], [pt1[1], pt2[1]], 'b-', linewidth=3)




            # Current rectangle: green.
            rect = np.array(rectangles[i] + [rectangles[i][0]])
            ax.plot(rect[:, 0], rect[:, 1], 'g--', linewidth=0.5, label='TCP Box')

            # Current grid points: red.
            grids = np.array(grid_points[i])
            if grid_points is not None:
                ax.plot(grids[:, 0], grids[:, 1], 'rx', markersize=4, label='Test Grid Points')

            ax.set_xlim(min_xy[0], max_xy[0])
            ax.set_ylim(min_xy[1], max_xy[1])
            ax.set_aspect('equal')
            ax.grid(True)
            _draw_inner_contours(ax, inner_contour_rings, used_labels)
            ax.legend()
            plt.tight_layout()
            pair_path = imageout_basepath/f"pair {iii+1}"
            pair_path.mkdir(exist_ok=True, parents=True)
            if save_image:
                plt.savefig(pair_path/f"possible_tcp_edge_{i+1}.svg", format='svg', bbox_inches='tight')
            if (not no_image and show_plt_TCP_each_edge):
                plt.show()
            else:
                plt.close()

    if save_image == True or (not no_image and show_plt_TCP_each_edge) == True:
        highlight_segment_rect_grid(contour_segments_2d_p2, tcp_box, test_grid_points, inner_contour_rings=_p2_inner_rings)


    # Show Gripper Bounding Box
    def create_gripper_bounding_box(grid_points, segments_2d):

        all_shapes = []
        segment_directions = [pt2 - pt1 for pt1, pt2 in segments_2d]
        segment_middle_point = [(pt1 + pt2) / 2 for pt1, pt2 in segments_2d]

        for pts, seg_dir,mid in zip(grid_points, segment_directions,segment_middle_point):
            seg_dir = seg_dir / np.linalg.norm(seg_dir)
            normal = np.array([-seg_dir[1], seg_dir[0]])

            
            segment_shapes = []

            for pt in pts:
                pt = np.array(pt)
                grid_edge_distance = np.dot(mid - pt, normal)

                rectangles = []
                
                #Safespace Finger front
                center1 = pt - normal * (x_pg + rj)
                p11 = center1 + seg_dir * (e_pg + 2*(i_pg + rj))/2 
                p12 = center1 + seg_dir * (e_pg + 2*(i_pg + rj))/2 + normal * (x_pg + rj)
                p13 = center1 - seg_dir * (e_pg + 2*(i_pg + rj))/2 + normal * (x_pg + rj)
                p14 = center1 - seg_dir * (e_pg + 2*(i_pg + rj))/2
                rectangles.append([p11, p12, p13, p14])

                #Finger length
                center2 = pt 
                p21 = center2 + seg_dir * (e_pg + 2*(i_pg + rj))/2
                p22 = center2 + seg_dir * (e_pg + 2*(i_pg + rj))/2 + normal * (b_pg + c_pg + rj)
                p23 = center2 - seg_dir * (e_pg + 2*(i_pg + rj))/2 + normal * (b_pg + c_pg + rj)
                p24 = center2 - seg_dir * (e_pg + 2*(i_pg + rj))/2
                rectangles.append([p21, p22, p23, p24])

                #Gripper Base
                center3 = center2 + normal * (b_pg + c_pg + rj)
                p31 = center3 + seg_dir * (j_pg + 2*rj)/2 
                p32 = center3 + seg_dir * (j_pg + 2*rj)/2 + normal * (d_pg + t_pg + u_pg + rj)
                p33 = center3 - seg_dir * (j_pg + 2*rj)/2 + normal * (d_pg + t_pg + u_pg + rj)
                p34 = center3 - seg_dir * (j_pg + 2*rj)/2
                rectangles.append([p31, p32, p33, p34])

                #Robot Arm
                center4 = center3 + normal * (d_pg + t_pg + u_pg + rj)
                p41 = center4 + seg_dir * (rd + re + 2*rj)/2
                p42 = center4 - seg_dir * (rd + re + 2*rj)/2
                p43 = center4 - seg_dir * (rd + re + 2*rj)/2 + normal * (rc + rf + 2*rj)
                p44 = center4 + seg_dir * (rd + re + 2*rj)/2 + normal * (rc + rf + 2*rj)
                rectangles.append([p41, p42, p43, p44])

                #Gripper Area
                center5 = pt
                p51 = center5 + seg_dir * (z_pg - 2*rj)/2
                p52 = center5 + seg_dir * (z_pg - 2*rj)/2 + normal * (b_pg - 2*rj)   
                p53 = center5 - seg_dir * (z_pg - 2*rj)/2 + normal * (b_pg - 2*rj)
                p54 = center5 - seg_dir * (z_pg - 2*rj)/2
                rectangles.append([p51, p52, p53, p54])

                #Robot Back Space
                center6 = center4 + normal * (rc + rf + 2*rj)
                p61 = center6 + seg_dir * (rd + re + 2*rj)/2
                p62 = center6 + seg_dir * (rd + re + 2*rj)/2 + normal * (grid_edge_distance + x_pg + rj)
                p63 = center6 - seg_dir * (rd + re + 2*rj)/2 + normal * (grid_edge_distance + x_pg + rj)
                p64 = center6 - seg_dir * (rd + re + 2*rj)/2
                rectangles.append([p61, p62, p63, p64])

                segment_shapes.append({
                    'point': pt,
                    'rectangles': rectangles
                })

            all_shapes.append(segment_shapes)

        return all_shapes


    def show_gripper_bounding_box(segments_2d, tcp_box, shapes, inner_contour_rings=None):
        all_pts = [pt for seg in segments_2d for pt in seg]
        bounds = np.array(all_pts)
        min_xy = bounds.min(axis=0) - plt_graphic_padding
        max_xy = bounds.max(axis=0) + plt_graphic_padding
    
        for i, segment_shape in enumerate(shapes):
            for j, shape in enumerate(segment_shape):
                fig, ax = plt.subplots(figsize=(8, 8))#Bounding Boxes
                ax.set_title(f"Edge {i+1}, Point {j+1}: Bounding Boxes")

                used_labels = set()

                # All line segments (blue).
                lbl = 'Outer Edges of Plane2'
                for pt1, pt2 in segments_2d:
                    if lbl not in used_labels:
                        ax.plot([pt1[0], pt2[0]], [pt1[1], pt2[1]], 'b-', linewidth=1, label=lbl)
                        used_labels.add(lbl)
                    else:
                        ax.plot([pt1[0], pt2[0]], [pt1[1], pt2[1]], 'b-', linewidth=1)

                # Initial rectangle (TCP box).
                rect = np.array(tcp_box[i] + [tcp_box[i][0]])
                lbl = 'TCP Box'
                if lbl not in used_labels:
                    ax.plot(rect[:, 0], rect[:, 1], 'g--', linewidth=0.5, label=lbl)
                    used_labels.add(lbl)
                else:
                    ax.plot(rect[:, 0], rect[:, 1], 'g--', linewidth=0.5)

                # Current test point.
                pt = shape['point']
                lbl = 'Test Point'
                if lbl not in used_labels:
                    ax.plot(pt[0], pt[1], 'ro', markersize=4, label=lbl)
                    used_labels.add(lbl)
                else:
                    ax.plot(pt[0], pt[1], 'ro', markersize=4)

                # Six rectangles.
                colors = ['red', 'purple', 'orange','deepskyblue','yellow','limegreen']
                Box_label = ['Finger Clearence Box', 'Finger Box', 'Finger Base Box', 'Robot Arm Box','Gripper Area Box','Robot Back Clearence Box']
                for k, rect in enumerate(shape['rectangles']):
                    poly = np.array(rect + [rect[0]])
                    lbl = Box_label[k]
                    if lbl not in used_labels:
                        ax.plot(poly[:, 0], poly[:, 1], color=colors[k], linewidth=1.5, label=lbl)
                        used_labels.add(lbl)
                    else:
                        ax.plot(poly[:, 0], poly[:, 1], color=colors[k], linewidth=1.5)


                ax.set_xlim(min_xy[0], max_xy[0])
                ax.set_ylim(min_xy[1], max_xy[1])
                ax.set_aspect('equal')
                ax.grid(True)
                _draw_inner_contours(ax, inner_contour_rings, used_labels)
                ax.legend()
                plt.tight_layout()
                plt.show()

    points_and_gripper_bounding_box = create_gripper_bounding_box(test_grid_points, contour_segments_2d_p2)

    if no_image == False:
        if show_plt_bounding_boxes == True:
            show_gripper_bounding_box(contour_segments_2d_p2,tcp_box,points_and_gripper_bounding_box,inner_contour_rings=_p2_inner_rings)



    #************************ Project P134 to 2 (CV2) **********************


    #*******************************





    def get_plane_contour_polygon(pcd, dir1, dir2, center, plane_name=""):
        """
        P1 / P3 / P4 / P5 entry point: just run the shared detection pipeline once
        and build the polygon-with-hole view from it.
        """
        detection = _detect_plane_2d_contours(pcd, dir1, dir2, center)
        return _build_plane_polygons(detection, pcd, plane_name)

    plane_contour_polygon_list = [
        get_plane_contour_polygon(overlap_pcd,dir1,dir2,center,'Plane 1'),
        plane2_polygons,  # Already computed once inside extract_and_visualize_contour_segments_with_normals; no need to redetect P2's contours.
        get_plane_contour_polygon(proj_pcd_p3,dir1,dir2,center,'Plane 3'),
        get_plane_contour_polygon(proj_pcd_p4,dir1,dir2,center,'Plane 4'),
        get_plane_contour_polygon(proj_pcd_p5,dir1,dir2,center,'Plane 5'),
        ]

    # plane_contour_polygon_list = [polygon_p1,polygon_p2,polygon_p3,polygon_p4]

    #********************** Loop Find feasible TCP *********************

    GRID_SIZE = 1e-9

    def _clean_geom(geom, name=""):
        if geom.is_empty:
            return geom

        g = geom

        # 1) Repair ineffective
        if not g.is_valid:
            g = make_valid(g)
        if not g.is_valid:
            g = g.buffer(0)

        # 2) set_precision：It's safer to place it later.
        try:
            g = set_precision(g, GRID_SIZE)
        except Exception as e:
            print(f"[WARN] set_precision failed on '{name}': {e}")

        # 3) merge
        try:
            if hasattr(g, "geoms"):
                g = unary_union(g)
        except Exception:
            pass

        # 4) Final validity check
        if not g.is_valid:
            msg = explain_validity(g)
            print(f"[WARN] Geometry '{name}' still invalid after cleaning: {msg}")

        return g

    def _safe_intersection_area(a, b):
        """
        Enable grid_size (snap rounding) during intersection, and fall back to buffer(0) when an exception occurs.
        """
        try:
            return a.intersection(b, grid_size=GRID_SIZE).area
        except GEOSException:
            a2 = _clean_geom(a, "A@fallback")
            b2 = _clean_geom(b, "B@fallback")
            return a2.intersection(b2, grid_size=GRID_SIZE).area
        

    def find_feasible_tcp(plane_contour_polygon_list,all_shapes):

        filtered_shapes = []
        feasible_points_on_edge = []
        intersection_areas_on_edge = []
        candidate_points_on_edge = []
        min_area = 0.15 * (z_pg-2*rj) * (b_pg-2*rj)

        # First convert each item in plane_contour_polygon_list into a list of polygons and clean them.
        for i in range(5):
            lst = plane_contour_polygon_list[i]
            if isinstance(lst, Polygon):
                plane_contour_polygon_list[i] = [lst]
            # Clean each polygon and apply precision settings.
            plane_contour_polygon_list[i] = [_clean_geom(p, f"plane_poly_{i}") for p in plane_contour_polygon_list[i]]

        poly_p1_list = plane_contour_polygon_list[0]
        poly_p2_list = plane_contour_polygon_list[1]
        poly_p3_list = plane_contour_polygon_list[2]
        poly_p4_list = plane_contour_polygon_list[3]        
        poly_p5_list = plane_contour_polygon_list[4]


        for segment_shapes in all_shapes:
            filtered_segment = []
            feasible_point = []
            intersection_areas = []
            candidate_point = []
            for shape in segment_shapes:
                pt = shape['point']
                rectangles = shape['rectangles']
                
                point_geom = Point(pt)
                rect1_geom = Polygon(rectangles[0])  # Finger tip Safe Space
                rect2_geom = Polygon(rectangles[1])  # Finger length
                rect3_geom = Polygon(rectangles[2])  # Gripper Base
                rect4_geom = Polygon(rectangles[3])  # Robot arm 
                rect5_geom = Polygon(rectangles[4])  # Gripper Area
                rect6_geom = Polygon(rectangles[5])  # Robot back sapace Box

                total_intersection_areas = sum(poly.intersection(rect5_geom).area for poly in poly_p1_list)

                # condition_1 = any(poly.contains(p oint_geom) for poly in poly_0_list)
                condition_1 = total_intersection_areas > min_area 
                condition_2 = all(
                    not poly.intersects(rect3_geom) and not poly.intersects(rect4_geom)
                    for poly in poly_p2_list
                )
                condition_3 = all(
                    not poly.intersects(rect1_geom) and
                    not poly.intersects(rect2_geom)
                    for poly in poly_p3_list 
                )
                condition_4 = all(
                    not poly.intersects(rect3_geom) and
                    not poly.intersects(rect4_geom)
                    for poly in poly_p4_list
                )
                condition_5 = all(not poly.intersects(rect4_geom) for poly in poly_p5_list)

                if  condition_1 and condition_2 and condition_3 and condition_4 and condition_5:
                    filtered_segment.append(shape)
                    feasible_point.append(pt)
                    intersection_areas.append(total_intersection_areas)
                
                candidate_point.append(pt)
                # if condition_1 :
                #     filtered_segment.append(shape)
                #     point_1.append(pt)                

            filtered_shapes.append(filtered_segment)
            feasible_points_on_edge.append(feasible_point)
            intersection_areas_on_edge.append(intersection_areas)
            candidate_points_on_edge.append(candidate_point)
        return filtered_shapes,feasible_points_on_edge,intersection_areas_on_edge,candidate_points_on_edge

    feasible_TCP_and_shapes,feasible_TCP_2d,intersection_areas,candidate_TCP_2d = find_feasible_tcp(plane_contour_polygon_list,points_and_gripper_bounding_box)
    # feasible_TCP_2d = [shape['point'] for segment in feasible_TCP_and_shapes for shape in segment]



    #*********************** Ranking function ******************************************

    def project_pts_to_3d(points, center, dir1, dir2): 
        center = np.asarray(center, dtype=float)
        dir1 = np.asarray(dir1, dtype=float)
        dir2 = np.asarray(dir2, dtype=float)  

        basis = np.vstack([dir1, dir2])     
        points_list_3d = []

        for pts in points:
            if not pts:
                points_list_3d.append(np.array([],dtype=float))
                continue
            uv = np.asarray(pts, dtype=float).reshape(-1, 2)
            p_3d = center + uv @ basis
            points_list_3d.append(p_3d) 

        return points_list_3d

    def get_area_score(intersection_areas):
        area_scores = []
        max_area = max((z_pg - 2*rj) * (b_pg - 2*rj), 1e-9)  # avoid 0
        for areas in intersection_areas:
            arr = np.asarray(areas, dtype=float)
            s = (arr - 0.15*max_area) / (0.85*max_area)
            s = np.clip(s, 0.0, 1.0)  # a<0.15*max -> 0；a>max -> 1
            area_scores.append(s)
        return area_scores


    def get_center_score(TCP_points, center_pcd, max_distance):

        center_pcd = np.asarray(center_pcd, dtype=float)
        TCP_points_dist = []
        
        for pts in TCP_points:
            if pts.size == 0:
                TCP_points_dist.append(np.array([],dtype=float))
                continue
            dist = np.linalg.norm(pts - center_pcd,axis=1)
            TCP_points_dist.append(dist)

        # # Initial Version: use the farest distance within the feasible TCP box as max 
        # non_empty = [d for d in TCP_points_dist if d.size > 0]
        # if len(non_empty) == 0:
        #     return [d.copy() for d in TCP_points_dist]
        # max_dist = np.max([np.max(d) for d in non_empty])

        # if max_dist == 0:
        # # all points are the same at center of mass -> give all 1.0
        #     TCP_dist_scores = [np.ones_like(d) for d in TCP_points_dist]
        # else:
        #     TCP_dist_scores = [(1.0 - d/max_dist) for d in TCP_points_dist]


        # Final Version: use the max_distance of whole pcd as max distance
        if max_distance <= 0:

            TCP_dist_scores = [np.ones_like(d) for d in TCP_points_dist]
            return TCP_dist_scores

        TCP_dist_scores = [np.clip(1.0 - d / max_distance, 0.0, 1.0) for d in TCP_points_dist]

        return TCP_dist_scores




    def rank_feasible_tcp(feasible_TCP_2d, intersection_areas,w1,w2):

        # 2D mean of original_points
        means = np.mean(original_points, axis=0)
        # find the farest distance to mean point within the whole pcd
        distances = np.linalg.norm(original_points - means, axis=1)
        max_idx = np.argmax(distances)
        max_distance = distances[max_idx]


        area_scores = get_area_score(intersection_areas)                      # list[list[float]]

        tcp_3d = project_pts_to_3d(feasible_TCP_2d, center, dir1, dir2)
        center_scores = get_center_score(tcp_3d, means, max_distance)  # list[list[float]]

        ranked = []
        for c_seg, a_seg in zip(center_scores, area_scores):
            # ensure same length per segment (should match by construction)
            m = min(len(c_seg), len(a_seg))
            ranked.append([w1 * c_seg[k] + w2 * a_seg[k] for k in range(m)])
        return ranked,tcp_3d
    feasible_TCP_rank,feasible_TCP_3d = rank_feasible_tcp(feasible_TCP_2d,intersection_areas,gss_w1,gss_w2)

    candidate_TCP_3d = project_pts_to_3d(candidate_TCP_2d, center, dir1, dir2)
    candidate_TCP_all_pairs.append(candidate_TCP_3d)



#*********************** Store the result **************

    grasp_elements = [feasible_TCP_3d,intersection_areas,feasible_TCP_rank]

    for i, inner_lists in enumerate(zip(*grasp_elements)):
        for j, items in enumerate(zip(*inner_lists)):
            rec = GraspData(
                TCP = items[0], # feasible_TCP_3d[i][j]
                pose = grasp_pose[i], # grasp_pose[i]
                contact_area = items[1], # intersection_areas[i][j]
                grasp_width = dist_plane,
                score = items[2], # feasible_TCP_rank[i][j]
                pair_num = iii+1,
                edge_num = i+1,
                point_num = j+1
                )
            store.add(rec)

#*********************************************************************************


    def highlight_feasible_tcp(TCP_points,TCP_rank, segments_2d, tcp_box, inner_contour_rings=None):
        """
        Iterate through each group of shapes in filtered_shapes and highlight them one by one:

        - All line segments (blue)

        - The TCP rectangle for the current point (green)

        - The current point location (red)

        Parameters:
        - filtered_shapes: List[List[dict]], each shape contains point and rectangles
        - segments_2d: list of 2D line segments [(pt1, pt2), ...]
        - rectangles_per_shape: same structure as filtered_shapes, used to provide the TCP box (rectangles[0])
        """

        all_pts = np.array(list(itertools.chain.from_iterable(segments_2d)))
        min_xy = all_pts.min(axis=0) - plt_graphic_padding
        max_xy = all_pts.max(axis=0) + plt_graphic_padding

        for i,pt in enumerate(TCP_points):
            fig, ax = plt.subplots(figsize=(8, 8)) # feasible each edge
            ax.set_title(f"Edge {i+1}: Feasible TCP and TCP Box")

            # All line segments: blue
            used_labels = set()
            lbl = 'Outer Coutour of Plane 2'
            for j, (pt1, pt2) in enumerate(segments_2d):

                if j == i:
                    mid = (pt1 + pt2) / 2
                    vec_12 = pt2 - pt1
                    vec_12 = vec_12 / np.linalg.norm(vec_12)
                    normal_clockwise_90 = [vec_12[1], -vec_12[0]]
                    normal_clockwise_90 = normal_clockwise_90 / np.linalg.norm(normal_clockwise_90)

                    #parallel symbol
                    # start_point_line = mid - normal_clockwise_90 * 0.026
                    # end_point_line = start_point_line + normal_clockwise_90 * 0.015
                    # end_point_base1 = end_point_line + vec_12 * 0.005
                    # end_point_base2 = end_point_line - vec_12 * 0.005
                    # end_point_finger1 = end_point_base1 + normal_clockwise_90 * 0.008
                    # end_point_finger2 = end_point_base2 + normal_clockwise_90 * 0.008

                    #tilt symbol
                    start_point_line = mid - normal_clockwise_90 * tilt_symbol_start_dist
                    end_point_line = start_point_line + normal_clockwise_90 * tilt_symbol_handle_length
                    end_point_base1 = end_point_line + vec_12 * tilt_symbol_finger_width_half - normal_clockwise_90 * tilt_symbol_finger_width_half
                    end_point_base2 = end_point_line - vec_12 * tilt_symbol_finger_width_half + normal_clockwise_90 * tilt_symbol_finger_width_half
                    end_point_finger1 = end_point_base1 + normal_clockwise_90 * tilt_symbol_finger_end_length
                    end_point_finger2 = end_point_base2 + normal_clockwise_90 * tilt_symbol_finger_end_length

                    ax.plot([start_point_line[0], end_point_line[0]], [start_point_line[1], end_point_line[1]], 'm', linewidth=1.5,label='Gripper Direction')
                    ax.plot([end_point_base1[0], end_point_base2[0]], [end_point_base1[1], end_point_base2[1]], 'm', linewidth=1.5)
                    ax.plot([end_point_base1[0], end_point_finger1[0]], [end_point_base1[1], end_point_finger1[1]], 'm', linewidth=1.5)
                    ax.plot([end_point_base2[0], end_point_finger2[0]], [end_point_base2[1], end_point_finger2[1]], 'm', linewidth=1.5)
                    
                if lbl not in used_labels:
                    ax.plot([pt1[0], pt2[0]], [pt1[1], pt2[1]], 'b-', linewidth=3, label=lbl)
                    used_labels.add(lbl)
                else:
                    ax.plot([pt1[0], pt2[0]], [pt1[1], pt2[1]], 'b-', linewidth=3)

            # Current point: green point
            pt = np.array(pt).reshape(-1, 2)
            # print(pt.shape)          
            # print(pt.ndim)
            if pt.size:
                scores_i = np.array(TCP_rank[i], dtype=float)  # (Ni,)
                if scores_i.shape[0] != pt.shape[0]:
                    print(f"[warn] edge {i}: #scores({scores_i.shape[0]}) != #pts({pt.shape[0]})")
                    # Optional: truncate or skip
                    m = min(scores_i.shape[0], pt.shape[0])
                    pt = pt[:m]
                    scores_i = scores_i[:m]
                # ax.plot(pt[:,0], pt[:,1],linestyle='None', marker='x', color='lime', label='Feasible TCP Point')
                scatter = plt.scatter(pt[:, 0], pt[:, 1], c=scores_i, cmap='RdYlGn',vmin=0, vmax=1, s=5, label='Feasible TCP Point')
                # cbar = plt.colorbar(scatter, ax=ax, label='Score', fraction=0.046, pad=0.04)
                # cbar.set_ticks([0.0, 0.5, 1.0])  
                # cbar.set_ticklabels([f"0.0", f"0.5", f"1.0"])

                divider = make_axes_locatable(ax)
                cax = divider.append_axes("right", size="5%", pad=0.3)
                cbar = plt.colorbar(scatter, cax=cax, label='Score')
                cbar.set_ticks([0.0, 0.5, 1.0])
                cbar.set_ticklabels(["0.0", "0.5", "1.0"])
            else:
                print("No feasible TCP point found!")
                ax.plot([], [],linestyle='None', marker='x', color='lime', label='Feasible TCP Point')
                # scatter = plt.scatter([], [], c=scores_i, cmap='RdYlGn', s=100, label='Feasible TCP Point')   

            # Current rectangle (rectangles[0]): green dashed box
            rect = np.array(tcp_box[i] + [tcp_box[i][0]])  # Closed polygon
            ax.plot(rect[:, 0], rect[:, 1], 'g--', linewidth=0.5, label='TCP Box')


            ax.set_xlim(min_xy[0], max_xy[0])
            ax.set_ylim(min_xy[1], max_xy[1])
            ax.set_aspect('equal')
            ax.grid(True)
            _draw_inner_contours(ax, inner_contour_rings, used_labels)  
            ax.legend()
            plt.tight_layout()
            pair_path = imageout_basepath/f"pair {iii+1}"
            pair_path.mkdir(exist_ok=True, parents=True)
            if save_image:
                plt.savefig(pair_path/f"feasible_tcp_edge_{i+1}.svg", format='svg', bbox_inches='tight')
            if (not no_image and show_feasible_each_edge):
                plt.show()
            else:
                plt.close()

    if save_image == True or (not no_image and show_feasible_each_edge) == True:
            highlight_feasible_tcp(feasible_TCP_2d,feasible_TCP_rank,contour_segments_2d_p2,tcp_box, inner_contour_rings=_p2_inner_rings)



    def highlight_feasible_all_tcp(TCP_points, TCP_rank, segments_2d, tcp_box, inner_contour_rings=None):
        """
        Display:
        - All line segments (blue)
        - Feasible TCP points on each edge (colored by score)
        - TCP rectangle for each edge (green dashed line)

        Parameters:
        - TCP_points: List[np.ndarray(Ni,2)], TCP point set for each edge
        - TCP_rank: List[np.ndarray(Ni,)], scores corresponding to TCP_points
        - segments_2d: list of (pt1, pt2), two endpoints of each edge
        - tcp_box: list of np.ndarray(4,2) or (M,2), four vertices of each TCP box (in order)
        """

        # Collect all points to define the coordinate range.
        all_xy = []
        for (pt1, pt2) in segments_2d:
            all_xy.extend([pt1, pt2])
        for pts in TCP_points:
            if pts is not None and len(pts) > 0:
                all_xy.extend(list(np.asarray(pts)))
        for rect in tcp_box:
            r = np.asarray(rect)
            if r.ndim == 2 and r.shape[0] >= 3:
                all_xy.extend(list(r))

        all_xy = np.asarray(all_xy) if len(all_xy) else np.zeros((1,2))
        min_xy = all_xy.min(axis=0) - plt_graphic_padding
        max_xy = all_xy.max(axis=0) + plt_graphic_padding

        fig, ax = plt.subplots(figsize=(8, 6)) # feasible all tcp
        ax.set_title("All Feasible TCP and TCP Box")

        used_labels = set()
        scatter_handle = None
        cbar = None

        # Iterate over all four lists in sync to avoid index mismatch.
        for (pt1, pt2), tcp, rect, scores in zip(segments_2d, TCP_points, tcp_box, TCP_rank):
            # 1) Line segment – blue
            lbl = 'Outer Contour of Plane 2'
            if lbl not in used_labels:
                ax.plot([pt1[0], pt2[0]], [pt1[1], pt2[1]], 'b-', linewidth=3, label=lbl)
                used_labels.add(lbl)
            else:
                ax.plot([pt1[0], pt2[0]], [pt1[1], pt2[1]], 'b-', linewidth=3)

            # 2) TCP points – colored by score
            tcp = np.asarray(tcp) if tcp is not None else np.empty((0,2))
            scores = np.asarray(scores, dtype=float) if scores is not None else np.empty((0,))
            if tcp.ndim == 1 and tcp.size == 2:
                tcp = tcp.reshape(1, 2)

            if tcp.size > 0:
                # Alignment Length
                m = min(len(tcp), len(scores))
                if m == 0:
                    pass
                else:
                    tcp = tcp[:m]
                    scores = scores[:m]
                    # Try setting colorbar limits based on data range.
                    # vmin = np.nanmin(scores) if np.isfinite(scores).any() else 0.0
                    # vmax = np.nanmax(scores) if np.isfinite(scores).any() else 1.0
                    vmin = 0.0
                    vmax = 1.0
                    if vmin == vmax:
                        vmin, vmax = vmax - 1.0, vmax + 1.0

                    lbl = 'Feasible TCP Point'
                    scatter_handle = ax.scatter(tcp[:, 0], tcp[:, 1],
                                                c=scores, cmap='RdYlGn',marker='x',
                                                vmin=vmin, vmax=vmax, s=20,
                                                label=(lbl if 'Feasible TCP Point' not in used_labels else None))
                    if 'Feasible TCP Point' not in used_labels:
                        used_labels.add('Feasible TCP Point')

                    if cbar is None and scatter_handle is not None:
                        # cbar = plt.colorbar(scatter_handle, ax=ax, label='Score', fraction=0.046, pad=0.04)
                        divider = make_axes_locatable(ax)
                        cax = divider.append_axes("right", size="5%", pad=0.3)
                        cbar = plt.colorbar(scatter_handle, cax=cax, label='Score')
                        cbar.set_ticks([0.0, 0.5, 1.0])
                        cbar.set_ticklabels(["0.0", "0.5", "1.0"])


            # 3) TCP rectangle – green dashed line (closed)
            rect = np.asarray(rect)
            if rect.ndim == 2 and rect.shape[0] >= 3:
                rect_closed = np.vstack([rect, rect[0]])
                lbl = 'TCP Box'
                if lbl not in used_labels:
                    ax.plot(rect_closed[:, 0], rect_closed[:, 1], 'g--', linewidth=0.5, label=lbl)
                    used_labels.add(lbl)
                else:
                    ax.plot(rect_closed[:, 0], rect_closed[:, 1], 'g--', linewidth=0.5)

        ax.set_xlim(min_xy[0], max_xy[0])
        ax.set_ylim(min_xy[1], max_xy[1])
        ax.set_aspect('equal')
        ax.grid(True, linestyle='--', alpha=0.4)
        _draw_inner_contours(ax, inner_contour_rings, used_labels)
        ax.legend(loc='best')
        plt.tight_layout()
        pair_path = imageout_basepath/f"pair {iii+1}"
        pair_path.mkdir(exist_ok=True, parents=True)
        if save_image:
            plt.savefig(pair_path/f"feasible_tcp_all_2d.svg", format='svg', bbox_inches='tight')
        if (not no_image and show_all_feasbile_in_2d):
            plt.show()
        else:
            plt.close()

    if save_image == True or (not no_image and show_all_feasbile_in_2d) == True:
            highlight_feasible_all_tcp(feasible_TCP_2d,feasible_TCP_rank,contour_segments_2d_p2,tcp_box, inner_contour_rings=_p2_inner_rings)


    def show_feasible_tcp_in_3d(TCP_3d, TCP_rank, segments_3d_para, pcd_pa, pcd_pb, pcd):
        
        # for pts in TCP_3d:
        #     if len(pts) == 0:
        #         print("No feasible TCP found to show in 3D!")
        #         return
        #     else:
        #         continue

        line_segments_3d,line_indices = segments_3d_para

        # Construct all line segments.
        if len(line_indices) == 0:
            print("No segments to visualize (the contour may be too small or overly simplified)")
            return
        
        blue = np.array([[0, 0, 1] for _ in range(len(TCP_3d))], dtype=float)

        line_set = o3d.geometry.LineSet()
        line_set.points = o3d.utility.Vector3dVector(np.asarray(line_segments_3d, dtype=float))
        line_set.lines = o3d.utility.Vector2iVector(np.asarray(line_indices, dtype=np.int32))
        line_set.colors = o3d.utility.Vector3dVector(blue)


        TCP_3d_flat = [p for group in TCP_3d for p in group]
        TCP_rank_flat = [r for group in TCP_rank for r in group]

        colors = plt.cm.RdYlGn(np.asarray(TCP_rank_flat, dtype=float))[:, :3]

        pcd_tcp = o3d.geometry.PointCloud()
        pcd_tcp.points = o3d.utility.Vector3dVector(TCP_3d_flat)
        pcd_tcp.colors = o3d.utility.Vector3dVector(colors)


        o3d.visualization.draw_geometries([pcd_tcp, pcd_pa, line_set],window_name='Feasible TCP in 3D with PA and PB')

        o3d.visualization.draw_geometries([pcd_tcp, pcd, line_set],window_name='Feasible TCP in 3D wit assembled part')


    if no_image == False:
        if show_feasible_with_P_and_pcd == True:
            show_feasible_tcp_in_3d(feasible_TCP_3d, feasible_TCP_rank, contour_segments_3d_p2_para, pcd_orig_i, pcd_orig_j, pcd)

#*************************** Grasp data store and best point search **************************
npz_outpath2 = Path(str(npz_outpath2).replace("candidate",f"candidate_of_{len(paired_planes)}_pairs"))

store.save_npz(npz_outpath)
store.save_csv(csv_outpath)

np.save(npz_outpath2, np.array(candidate_TCP_all_pairs, dtype=object))


best_record = store.best()
if best_record == None:
    print("===== No feasible Point Found =====")
else:
    print("===== Global Best Grasp =====")
    print("Score:", best_record.score)
    print("Pair / Edge / Point:", best_record.pair_num, best_record.edge_num, best_record.point_num)
    print("TCP:", best_record.TCP)
    print("Pose:", best_record.pose)
    print("Contact Area:", best_record.contact_area)
    print("Grasp Width:", best_record.grasp_width)

    top_5_record = store.top_k(5)
    print("===== Top 5 Grasp =====")
    for i, record in enumerate(top_5_record):
        print(f"Top[{i+1}]Pair / Edge / Point:", record.pair_num, record.edge_num, record.point_num)

    worst_record = store.worst()
    print("===== Worst Grasp =====")
    print("Score:", worst_record.score)
    print("Pair / Edge / Point:", worst_record.pair_num, worst_record.edge_num, worst_record.point_num)
    print("TCP:", worst_record.TCP)
    print("Pose:", worst_record.pose)
    print("Contact Area:", worst_record.contact_area)
    print("Grasp Width:", worst_record.grasp_width)