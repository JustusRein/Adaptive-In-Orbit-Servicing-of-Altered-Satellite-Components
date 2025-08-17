import open3d as o3d
import numpy as np
import random
import copy
from math import acos, degrees
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from collections import Counter
import math

import alphashape
from shapely.geometry import Point, MultiPoint,Polygon,MultiLineString, MultiPolygon,LineString
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from scipy.spatial import KDTree,cKDTree
import cv2
import itertools

from scipy.spatial import ConvexHull, Delaunay
from shapely.ops import unary_union, polygonize

import logging, pathlib
import yaml


# -------------------- Step 0: 定义函数 ----------------
class parallel_gripper:
    def __init__(self):
        space = 0.005
        a_pg = 0.01 # Finger width
        w_pg = 0.3*space # Internal Safespace Finger width 
        v_pg = space # External Safespace Finger width 
        f_pg = 0.10 # Distance gripper open
        g_pg = 0.02 # Distance gripper close
        h_pg = 0.12 # Gripper base bottom width
        k_pg = space # Safespace Gripper base bottom width 
        q_pg = 0.08 # Gripper base top width
        r_pg = space # Safespace Gripper base top width

        # b_pg = 0.01 # TCP to Finger length end
        c_pg = 0.04 # TCP to (Safety space of Gripper)length end
        d_pg = space # Safespace Gripper length
        x_pg = space # Safespace Gripper end to rubber
        n_pg = d_pg + c_pg + x_pg # Finger length
        t_pg = 0.065 # Gripper base bottom length
        u_pg = 0.05 # Gripper base top length
        j_pg = c_pg + d_pg + t_pg + u_pg # Gripper length (TCP to Robot)
        s_pg = j_pg + x_pg # Total gripper length

        e_pg = 0.04 # Finger depth
        i_pg = space # Safespace finger depth
        l_pg = 0.06 # Gripper base bottom depth
        m_pg = space # Safespace gripper base bottom depth
        o_pg = 0.07 # Gripper base top  depth
        p_pg = space # Safespace gripper base top depth

        y_pg = l_pg/2 + m_pg if (l_pg/2 + m_pg) >  (o_pg/2 + p_pg) else (o_pg/2 + p_pg) # Gripper Bounding box depth


# MM_TO_M = 0.001

# def load_yaml_in_m(filepath):
#     """读取 YAML 文件，将所有数值从 mm 转成 m"""
#     with open(filepath, "r", encoding="utf-8") as f:
#         data_mm = yaml.safe_load(f)
    
#     # 转成 m（只处理数值型数据）
#     data_m = {
#         key: (value * MM_TO_M if isinstance(value, (int, float)) else value)
#         for key, value in data_mm.items()
#     }
#     return data_m

# # mm to m
# params = load_yaml_in_m(r"D:\Codecouldcode\099.MA_Hanyu\01_project\gripper_parameter\Franka.yaml")

with open(r"gripper_parameter\Franka.yaml", "r", encoding="utf-8") as f:
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
        raise ValueError("n_ref 必须是非零、有限向量")
    n_ref = n_ref / nr

    need_est = (not pcd.has_normals()) or (len(pcd.normals) != N)

    if need_est:
        if N < 3:
            # 点太少无法用邻域PCA估计，直接赋已知平面法向量
            pcd.normals = o3d.utility.Vector3dVector(np.repeat(n_ref[None, :], N, axis=0))
        else:
            # 半径自适应更稳（适应尺度变化）
            if radius is None:
                pts = np.asarray(pcd.points)
                diag = float(np.linalg.norm(pts.max(0) - pts.min(0)))
                radius = max(1e-9, 0.02 * diag)  # 经验值：对角线的 ~2%
            pcd.estimate_normals(
                o3d.geometry.KDTreeSearchParamHybrid(
                    radius=radius,
                    max_nn=min(max_nn, max(3, N-1))
                )
            )
            # 保险再检查一次，仍不匹配就直接赋值
            if (not pcd.has_normals()) or (len(pcd.normals) != N):
                pcd.normals = o3d.utility.Vector3dVector(np.repeat(n_ref[None, :], N, axis=0))

    # 统一方向，失败就兜底为直接赋值
    try:
        pcd.orient_normals_to_align_with_direction(n_ref)
    except RuntimeError:
        pcd.normals = o3d.utility.Vector3dVector(np.repeat(n_ref[None, :], N, axis=0))

    pcd.normalize_normals()

    normals = np.asarray(pcd.normals)
    cosang = np.clip(normals @ n_ref, -1.0, 1.0)
    # 若已经对齐，这里其实无需 abs；保留 abs 更稳
    mask = (np.abs(cosang) >= float(cos_th))
    out = pcd.select_by_index(np.where(mask)[0])
    return out, mask

def remove_pcd_outlier_statistical(pcd, neighbots=20,std_ratio=1.0):

    if len(pcd.points) == 0:
        print("Remove Outlier Error: Input point cloud is empty.")
        return pcd,None
    
    #statitical
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
    
    #radius
    # radius=0.005
    # nb_points=10
    # if isinstance(pcd, o3d.geometry.PointCloud):
    #     filtered,ind = pcd.remove_radius_outlier(nb_points=nb_points, radius=radius)
    #     return filtered,ind
    # elif isinstance(pcd,np.ndarray):
    #     pcloud = o3d.geometry.PointCloud()
    #     pcloud.points = o3d.utility.Vector3dVector(pcd)
    #     filtered,ind = pcloud.remove_radius_outlier(nb_points=nb_points, radius=radius)
    #     return np.asarray(filtered.points),ind
    # else:
    #     print("Error: Input type is not supported, neither 'PointCloud' nor 'np.ndarray'.")
    #     return None,None

    #adaptiv    
    # points = pcd
    # input_is_o3d = False
    # nb_neighbors=20
    # std_ratio=2.0
    # verbose=True
    # if isinstance(points, o3d.geometry.PointCloud):
    #     input_is_o3d = True
    #     points_np = np.asarray(points.points)
    # elif isinstance(points, np.ndarray):
    #     points_np = points
    # else:
    #     raise TypeError("Input must be o3d PointCloud or np.ndarray")

    # nbrs = NearestNeighbors(n_neighbors=nb_neighbors+1).fit(points_np)
    # distances, _ = nbrs.kneighbors(points_np)
    # avg_distances = np.mean(distances[:, 1:], axis=1)

    # dist_mean = np.mean(avg_distances)
    # dist_std = np.std(avg_distances)

    # threshold = dist_mean + std_ratio * dist_std
    # mask = avg_distances <= threshold

    # if verbose:
    #     print(f"Mean dist: {dist_mean:.6f}, Std: {dist_std:.6f}, Threshold: {threshold:.6f}")
    #     print(f"原始点数: {len(points_np)}, 保留点数: {np.sum(mask)}, 去除点数: {len(points_np)-np.sum(mask)}")

    # filtered_points = points_np[mask]

    # if input_is_o3d:
    #     filtered_pcd = o3d.geometry.PointCloud()
    #     filtered_pcd.points = o3d.utility.Vector3dVector(filtered_points)
    #     return filtered_pcd, mask
    # else:
    #     return filtered_points, mask

def remove_pcd_outlier_dbscan(pcd, eps=0.007, min_samples=20,min_cluster_ratio=0.02,verbose=True):
    if len(pcd.points) <= 500:
        pcd_null = o3d.geometry.PointCloud()
        pcd_null.points = o3d.utility.Vector3dVector([])
        return pcd_null,None
    else:
        return pcd,None
# def remove_pcd_outlier_dbscan(pcd, eps=0.007, min_samples=20,min_cluster_ratio=0.02,verbose=True):
#     #dbscan

#     # 兼容输入类型
#     input_is_o3d = False
#     if isinstance(pcd, o3d.geometry.PointCloud):
#         points = np.asarray(pcd.points)
#         input_is_o3d = True
#     elif isinstance(pcd, np.ndarray):
#         points = pcd
#     else:
#         raise TypeError("Input must be o3d.geometry.PointCloud or np.ndarray.")

#     # DBSCAN聚类
#     try:
#         clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(points)
#         labels = clustering.labels_

#         # labels=-1 为噪声点
#         unique, counts = np.unique(labels, return_counts=True)
#         cluster_sizes = dict(zip(unique, counts))

#         # 确定哪些聚类视为主体
#         total_points = len(points)
#         main_clusters = [label for label, size in cluster_sizes.items()
#                         if label != -1 and size >= min_cluster_ratio * total_points]

#         mask = np.isin(labels, main_clusters)

#         if verbose:
#             print("聚类数量:", len(unique)-1)
#             print("识别为主体的聚类:", main_clusters)
#             print(f"原始点数: {total_points}, 保留点数: {np.sum(mask)}, 去除点数: {total_points - np.sum(mask)}")

#         filtered_points = points[mask]
#     except:
#         logging.warning("DBSCAN 聚类失败，跳过")
#         mask = np.zeros(len(points), dtype=bool)
#         return pcd,mask
    
#     if input_is_o3d:
#         filtered_pcd = o3d.geometry.PointCloud()
#         filtered_pcd.points = o3d.utility.Vector3dVector(filtered_points)
#         filtered_pcd.colors= o3d.utility.Vector3dVector(np.array(pcd.colors)[mask])
#         return filtered_pcd, mask
#     else:
#         return filtered_points, mask

# **************************** Step 1: 读取点云+外法线估计 ****************************
normal_radius = 0.05
curvature_radius = 0.05
dbscan_eps = 0.1
dbscan_min_samples = 20
local_k = 20  # 用于局部一致性纠正的邻域点数

logging.basicConfig(level=logging.INFO, format="%(message)s")


def orient_normals_outward(
    pcd: o3d.geometry.PointCloud,
    radius_factor: float = 4.0,   # 法向搜索半径 = radius_factor * avg_nn_distance
    main_eps_deg: float = 20.0,   # DBSCAN 聚类角度阈值 (度)
    min_samples: int = 20,
    k_consistency: int = 30,
):
    """估计并统一外部法向"""
    if pcd.is_empty():
        raise ValueError("Point cloud is empty!")

    # 计算平均点间距，作为尺度
    dists = pcd.compute_nearest_neighbor_distance()
    avg_d  = np.mean(dists)
    n_radius = radius_factor * avg_d

    logging.info(f"Avg NN distance: {avg_d:.4f}, using normal radius {n_radius:.4f}")

    # 1) 估计法向
    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=n_radius, max_nn=50)
    )
    pcd.normalize_normals()

    # 2) 找主方向（可选）
    normals = np.asarray(pcd.normals)
    clustering = DBSCAN(
        eps=np.sin(np.deg2rad(main_eps_deg)) * 2,  # 余弦距离阈值近似
        min_samples=min_samples, metric="cosine"
    ).fit(normals)
    labels = clustering.labels_
    valid   = labels >= 0
    if not valid.any():
        logging.warning("DBSCAN 未找到簇，跳过全局翻转")
    else:
        counts  = Counter(labels[valid])
        main_lab = counts.most_common(1)[0][0]
        main_dir = normals[labels == main_lab].mean(0)
        main_dir /= np.linalg.norm(main_dir)
        # 向量化翻转
        dot = (normals * main_dir).sum(1)
        normals[dot < 0] *= -1
        pcd.normals = o3d.utility.Vector3dVector(normals)

    # 3) 局部一致性 (Open3D 自带)
    pcd.orient_normals_consistent_tangent_plane(k_consistency)
    return pcd


path = pathlib.Path(r"Test_part\Verification_examples\00_CubeSat_Assembly v3_sampled10k.pcd")
pcd = o3d.io.read_point_cloud(str(path))
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
# ---------------- Step 2: 初始化 ----------------
plane_indices_list = []   # 存放每个平面的原始索引
plane_colors = []         # 用于可视化
plane_models = []         # 平面模型
plane_normals = []        # 平面法向量

min_remaining_points = 100
min_points_per_plane = 50      # 小平面过滤阈值

distance_threshold = 0.001       # 拟合误差
max_planes = 100              # 最多检测几个平面（可调大）

rest_pcd = copy.deepcopy(pcd)

# ---------------- Step 3: 提取多个平面 ----------------
#*******************GPTjia***************************
# ===== [ADD] 工具函数：平面归一化 + 全局再标注 =====

def _normalize_plane(model):
    """把 ax+by+cz+d=0 的平面方程归一化到 ||[a,b,c]|| = 1，便于用“几何距离”阈值。"""
    n = np.asarray(model[:3], dtype=float)
    d = float(model[3])
    s = np.linalg.norm(n)
    if s == 0:
        return n, d
    return n / s, d / s

def _global_relabel_by_planes(pcd,
                              plane_models,
                              dist_thr,
                              tie_margin_ratio=0.3,
                              angle_thr_deg=15):
    """
    方法一：在“原始点云”上按已检测出的平面模型做一次全局再标注。
    - 允许边缘/拐角点多重归属（近似等距的多个平面都可归属）。
    - 可选用法向一致性（若 pcd 已有法向）；否则仅用距离。
    返回：
      plane_indices_list_full: [np.array(...), ...] 每个平面的原始索引
      membership_mask: (N,K) 的布尔矩阵，便于你可视化多重归属
    """
    P = np.asarray(pcd.points)
    N_points = len(P)
    K = len(plane_models)
    if N_points == 0 or K == 0:
        return [], None

    # 归一化每个平面到几何距离
    Ns, Ds = [], []
    for m in plane_models:
        n, d = _normalize_plane(m)
        Ns.append(n)
        Ds.append(d)
    Ns = np.asarray(Ns)            # (K,3)
    Ds = np.asarray(Ds)            # (K,)

    # 点到各平面几何距离：|x·n + d|
    dists = np.abs(P @ Ns.T + Ds[None, :])    # (N,K)

    # 基于距离的主判据
    labels = (dists <= dist_thr)

    # 可选：法向一致性（若已有法向），抑制跨面误标
    if angle_thr_deg is not None and pcd.has_normals():
        normals = np.asarray(pcd.normals)     # (N,3)
        cos_thr = np.cos(np.deg2rad(angle_thr_deg))
        ang_ok = (np.abs(normals @ Ns.T) >= cos_thr)   # (N,K)
        labels = labels & ang_ok

    # 允许边界点“近似等距”的多重归属（tie-margin）
    tie_margin = float(dist_thr) * float(tie_margin_ratio)
    min_d = dists.min(axis=1, keepdims=True)                 # (N,1)
    valid = (min_d <= (dist_thr + tie_margin))               # (N,1)
    near_tie = (dists - min_d) <= tie_margin                 # (N,K)
    labels = labels | (near_tie & valid)

    plane_indices_list_full = [np.where(labels[:, k])[0] for k in range(K)]
    return plane_indices_list_full, labels

#***************************************************

def correct_normal_direction_by_density(pcd, plane_indices, plane_normal):
        
        all_normals = np.asarray(pcd.normals)
        outer_avg = all_normals[plane_indices].mean(axis=0)
        outer_avg /= np.linalg.norm(outer_avg)

        dot = np.dot(outer_avg, plane_normal)
        angle = np.arccos(np.clip(dot, -1.0, 1.0)) * 180 / np.pi
        print(f"average point normal vs fitted normal angle: {angle:.2f}°")
        
        # 与原始平面法向比一比，看需不需要翻转
        if angle > 90 :
            plane_normal = -plane_normal  # 让它朝外 
        
        return plane_normal

for i in range(max_planes):

    print(f"loop{i}")

    if len(rest_pcd.points) < min_remaining_points:
        print("Not enough points left to extract more planes.")
        break  # 剩余点太少，停止提取

    plane_model, inliers = rest_pcd.segment_plane(
        distance_threshold=distance_threshold,
        ransac_n=3,
        num_iterations=1000)

    if len(inliers) < min_points_per_plane:
        print(f"Plane has only {len(inliers)} points, skipping.")
        break  # 太小，停止提取

    # 保存当前平面的点索引（是当前点云的 index，需要映射回原始点云）
    current_points = np.asarray(rest_pcd.points)
    current_indices = np.arange(len(current_points))
    original_idx = original_indices[inliers]
    plane_indices_list.append(original_idx)

    ##todo 去除被切穿点*****


# ***************************************************
    #保存平面模型与法线
    plane_models.append(plane_model) # 平面模型
    normal_vector = np.asarray(plane_model[0:3])
    normal_vector = normal_vector / np.linalg.norm(normal_vector)
    # normal_vector = correct_normal_direction_by_density(pcd, original_idx, normal_vector)
    plane_normals.append(normal_vector) # 平面法向量

    # 为当前平面着色（随机）
    color = [random.random(), random.random(), random.random()]
    plane_colors.append(color)

    # 剩余点云
    rest_pcd = rest_pcd.select_by_index(inliers, invert=True)
    original_indices = np.delete(original_indices, inliers)


# relabel_dist_thr = distance_threshold          # 可按需要放宽，如 1.2 * distance_threshold
# tie_margin_ratio = 0.3                         # 近似等距放宽比例（边界允许多归属）
# angle_thr_deg = 15                             # 若 pcd 有法向则启用；没有法向会自动忽略

# plane_indices_list_full, membership_mask = _global_relabel_by_planes(
#     pcd=pcd,
#     plane_models=plane_models,
#     dist_thr=relabel_dist_thr,
#     tie_margin_ratio=tie_margin_ratio,
#     angle_thr_deg=angle_thr_deg
# )

# # 用“完整索引”覆盖原本的 plane_indices_list，保持后续变量名不变
# plane_indices_list = plane_indices_list_full

# # （可选）按“再标注后的规模”重新过滤过小平面，保持颜色/模型/法向一致对齐
# keep_mask = [len(idx) >= min_points_per_plane for idx in plane_indices_list]
# if not all(keep_mask):
#     plane_indices_list = [idx for idx, keep in zip(plane_indices_list, keep_mask) if keep]
#     plane_models      = [m   for m,   keep in zip(plane_models,      keep_mask) if keep]
#     plane_normals     = [n   for n,   keep in zip(plane_normals,     keep_mask) if keep]
#     plane_colors      = [c   for c,   keep in zip(plane_colors,      keep_mask) if keep]

# # （可选）若你需要每个平面的完整子云
# plane_subclouds = [pcd.select_by_index(idx.tolist()) for idx in plane_indices_list]




# ---------------- Step 4: 可视化着色 ----------------
# 创建一个点云，给所有点上色
colored_pcd = copy.deepcopy(pcd)
colors = np.ones((len(original_points), 3)) * [0.5, 0.5, 0.5]  # 默认灰色

for indices, color in zip(plane_indices_list, plane_colors):
    colors[indices] = color  # 为每个平面的点赋色

colored_pcd.colors = o3d.utility.Vector3dVector(colors)

#创建法线显示
def create_normal_arrow(origin, normal, length=0.02, color=[1, 0, 0]):
    """
    创建一个法线箭头用于可视化
    origin: 起点（三维坐标）
    normal: 单位法向量
    length: 箭头长度
    color: RGB颜色
    """
    arrow = o3d.geometry.TriangleMesh.create_arrow(
        cylinder_radius=0.001,
        cone_radius=0.002,
        cylinder_height=length * 0.8,
        cone_height=length * 0.2
    )
    arrow.paint_uniform_color(color)

    # 构造旋转矩阵，使箭头从Z方向旋转到 normal
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




# ---------------- 可选：输出平面索引列表 ----------------
for i, indices in enumerate(plane_indices_list):
    print(f"Plane {i}: {len(indices)} points, indices example: {indices[:5]}")
    # o3d.visualization.draw_geometries([pcd.select_by_index(indices.tolist())], window_name=f"Plane {i} points", width=800, height=600)


#显示法线和平面
o3d.visualization.draw_geometries([colored_pcd] + arrow_list, window_name="Plane segmentation result and external normal",point_show_normal=False)

####################################################################


def is_parallel(v1, v2, angle_thresh_deg=5):
    cos_theta = np.clip(np.dot(v1, v2), -1.0, 1.0)
    angle = degrees(acos(abs(cos_theta)))  # abs保证±方向一致也算平行
    return angle <= angle_thresh_deg

unclustered = set(range(len(plane_normals)))
parallel_groups = []

while unclustered:
    idx = unclustered.pop()
    ref_normal = plane_normals[idx]

    current_group = [idx]

    to_remove = []
    for other in unclustered:
        if is_parallel(ref_normal, plane_normals[other]):
            current_group.append(other)
            to_remove.append(other)

    for i in to_remove:
        unclustered.remove(i)

    parallel_groups.append(current_group)

#颜色

colored_pcd = copy.deepcopy(pcd)
colors = np.ones((len(pcd.points), 3)) * [0.5, 0.5, 0.5]  # 灰色默认背景

group_colors = [[random.random(), random.random(), random.random()] for _ in parallel_groups]

for group_idx, group in enumerate(parallel_groups):
    color = group_colors[group_idx]
    for plane_idx in group:
        point_indices = plane_indices_list[plane_idx]
        colors[point_indices] = color

colored_pcd.colors = o3d.utility.Vector3dVector(colors)
o3d.visualization.draw_geometries([colored_pcd], window_name="Planeclustering result and external normal")

###############################################平面配对

#todo: 平面配对
def is_opposite_direction(idx_i, idx_j):

    pn1 = np.asarray(plane_normals[idx_i])
    pn2 = np.asarray(plane_normals[idx_j])

    point_indices_i = plane_indices_list[idx_i]
    point_indices_j = plane_indices_list[idx_j]
    plane_points_i = np.asarray(pcd.select_by_index(point_indices_i).points)
    plane_points_j = np.asarray(pcd.select_by_index(point_indices_j).points)
    pc1 = np.mean(plane_points_i, axis=0)
    pc2 = np.mean(plane_points_j, axis=0)

    c1c2 = pc2 - pc1 #中心连线 1->2

    dot0 = np.dot(pn1, pn2)
    dot1 = np.dot(c1c2, pn1) #外法线时应为-
    dot2 = np.dot(c1c2, pn2) #外法线时应为+


    if dot0 < 0 and dot1 < 0 and dot2 > 0:  # 方向相同
        return True
    else:   
        return False



paired_planes = []  # 所有配对

for group in parallel_groups:
    n = len(group)
    for i in range(n):
        for j in range(i + 1, n):  # 防止 (i,j) 和 (j,i) 重复
            idx_i = group[i]
            idx_j = group[j]

            n1 = plane_normals[idx_i]
            n2 = plane_normals[idx_j]

            # if is_opposite_direction(idx_i, idx_j):
            paired_planes.append((idx_i, idx_j))

print(f"\n\n=======================================\n========== Paired planes: {len(paired_planes)} ==========\n=======================================")

#可视化
for count, (i, j) in enumerate(paired_planes):
    # 创建新的颜色数组，初始为灰色
    colors = np.ones((len(pcd.points), 3)) * [0.6, 0.6, 0.6]

    # 为当前配对指定颜色
    color = [random.random(), random.random(), random.random()]
    for idx in plane_indices_list[i]:
        colors[idx] = color
    for idx in plane_indices_list[j]:
        colors[idx] = color

    # 创建新点云并赋色
    paired_pcd = copy.deepcopy(pcd)
    paired_pcd.colors = o3d.utility.Vector3dVector(colors)

    print(f"Show pair：{i} ↔ {j}")
    # o3d.visualization.draw_geometries([paired_pcd], window_name=f"Pair {count+1}: Plane {i} 和 Plane {j}",width=800,height=600,)

#---------------Find center plane---------------

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

    if dist_plane < g_pg:
        print("\n####################\nPlane pair distance is too small, skip\n####################\n")
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
    pcd_proj_i.paint_uniform_color([0, 1, 0])  # 绿色

    pcd_proj_j = o3d.geometry.PointCloud()
    pcd_proj_j.points = o3d.utility.Vector3dVector(projected_j_points)
    pcd_proj_j.paint_uniform_color([1, 0, 0])  # 红色

    pcd_orig_i = pcd.select_by_index(plane_indices_list[mmm])
    pcd_orig_j = pcd.select_by_index(plane_indices_list[nnn])

    pcd_orig_i.paint_uniform_color([0.7, 0.7, 0.7])  # 淡灰色
    pcd_orig_j.paint_uniform_color([0.7, 0.7, 0.7])

    o3d.visualization.draw_geometries([
        pcd.translate([0,0.001,0]),
        pcd_orig_i,
        pcd_orig_j,
        pcd_proj_i,
        pcd_proj_j
    ], window_name="Plane Projection", width=800, height=600)

    if input("Skip this pair? (y/n)") == "y":
        continue

    #**************************** Plane 1: Project planes and find overlap region ****************************

    def extract_overlap_region(proj_A, proj_B, threshold=0.001,remove = False):
        """
        从两个已投影的点云中提取重合区域，返回合并后的重合点云
        """

        dA = np.asarray(proj_A.compute_nearest_neighbor_distance())
        dB = np.asarray(proj_B.compute_nearest_neighbor_distance())
        print("median spacing A/B:", np.median(dA), np.median(dB))

        threshold = 1.2 * max(np.median(dA), np.median(dB))
        # 构建 KDTree
        kdtree_B = o3d.geometry.KDTreeFlann(proj_B)
        kdtree_A = o3d.geometry.KDTreeFlann(proj_A)

        points_A = np.asarray(proj_A.points)
        points_B = np.asarray(proj_B.points)

        # A 中那些有邻近 B 点的
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
            # B 中那些有邻近 A 点的
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
                # 合并重合区域的点
                overlap_points = A_keep if B_keep.size == 0 else (B_keep if A_keep.size == 0 else np.vstack([A_keep, B_keep]))

            if len(overlap_points) < 200:
                print("\n############################\nThere are less than 200 points here.\n############################\n")
                return None
            
            pcd_overlap = o3d.geometry.PointCloud()
            pcd_overlap.points = o3d.utility.Vector3dVector(overlap_points)
            pcd_overlap.paint_uniform_color([0, 1, 0])  # 绿色标记

            return pcd_overlap

    overlap_pcd_unfilter = extract_overlap_region(pcd_proj_i, pcd_proj_j, threshold=0.001)
    if overlap_pcd_unfilter is None:
        continue

    overlap_pcd,ind_p1 = filter_by_normal_orientation(overlap_pcd_unfilter,plane_normals[mmm])
    overlap_pcd,ind_p1 = remove_pcd_outlier_statistical(overlap_pcd_unfilter)
    projected_points_p1 = np.asarray(overlap_pcd.points)

    colors = np.ones((len(overlap_pcd_unfilter.points), 3)) * [1,1,0]
    colors[ind_p1,:] = [0,1,0]
    overlap_pcd_unfilter.colors = o3d.utility.Vector3dVector(colors)

    o3d.visualization.draw_geometries([overlap_pcd_unfilter.translate([0,0,0.00001]),pcd_orig_i,pcd_orig_j],window_name="Pair of Planes and Their Overlap Region")


    #**************************** Plane 2: Find points between planes ****************************
    def project_points_to_plane(points, plane_point, plane_normal):
        v = points - plane_point
        d = np.dot(v, plane_normal)
        return points - np.outer(d, plane_normal)

    def select_points_between_planes(pcd, center_i, center_j, plane_normal, margin=0.0015, include_planes=True):
        """
        筛选出完整点云中夹在两个平面之间的点
        """

        if isinstance(pcd, o3d.geometry.PointCloud):
            points = np.asarray(pcd.points)
        elif isinstance(pcd, np.ndarray):
            points = pcd
        # center_i = plane_i_pts.mean(axis=0)
        # center_j = plane_j_pts.mean(axis=0)

        # 平面间距离向量（方向需与法向一致）
        dist_vec = center_j - center_i
        dist_vec /= np.linalg.norm(dist_vec)
        
        # 投影每个点到法向上，得到相对两个平面的距离
        d_i = np.dot(points - center_i, plane_normal)
        d_j = np.dot(points - center_j, plane_normal)

        # 判断点是否在两个平面之间（允许一个 margin 容差）
        if include_planes:
            mask = (d_i * d_j <= 0) | (np.abs(d_i) <= margin) | (np.abs(d_j) <= margin)
        else:
            mask = (d_i * d_j < 0) & (np.abs(d_i) > margin) & (np.abs(d_j) > margin)

        points_between = points[mask]
        points_beside = points[~mask]
        return points_between,points_beside




    # 1. 筛选出在两个平面之间的点
    points_between_p2,points_beside = select_points_between_planes(pcd, center_i, center_j, plane_normals[mmm])

    # 2. 投影到中间平面
    projected_points_p2 = project_points_to_plane(points_between_p2, center_ij, plane_normals[mmm])

    # 3. 创建 PointCloud 对象
    proj_pcd_p2_unfilter = o3d.geometry.PointCloud()
    proj_pcd_p2_unfilter.points = o3d.utility.Vector3dVector(projected_points_p2)
    proj_pcd_p2_unfilter.paint_uniform_color([1, 0, 0])  # 红色

    proj_pcd_p2,ind_p2 = remove_pcd_outlier_dbscan(proj_pcd_p2_unfilter)
    projected_points_p2 = np.asarray(proj_pcd_p2.points)

    colors = np.ones((len(proj_pcd_p2_unfilter.points), 3)) * [1,1,0]
    colors[ind_p2,:] = [1,0,0]
    proj_pcd_p2_unfilter.colors = o3d.utility.Vector3dVector(colors)

    # 可视化所有内容
    o3d.visualization.draw_geometries([
        pcd_orig_j, pcd_orig_i, proj_pcd_p2_unfilter
        ,
    ],window_name="Pair of Planes and Projected Points Between Them")


    #**************************** Plane 3a,3b: find outside(finger) collision area ****************************

    #*P3a
    center_i_p3a = center_i + (a_pg + w_pg + v_pg) * (plane_normals[mmm]) * dist_dir_i
    center_j_p3a = center_j + (a_pg + w_pg + v_pg) * (plane_normals[nnn]) * dist_dir_j
    # center_i_p3 = center_ij + (0.02) * (plane_normals[mmm]) * dist_dir_i
    # center_j_p3 = center_ij + (0.02) * (plane_normals[nnn]) * dist_dir_j

    # 1. 筛选出在两个平面之间的点
    points_between_p3a_i,points_beside = select_points_between_planes(points_beside, center_i, center_i_p3a, plane_normals[mmm])
    points_between_p3a_j,points_beside = select_points_between_planes(points_beside, center_j, center_j_p3a, plane_normals[nnn])
    points_between_p3a = np.vstack((points_between_p3a_i, points_between_p3a_j))

    # 2. 投影到中间平面
    projected_points_p3a = project_points_to_plane(points_between_p3a, center_ij, plane_normals[mmm])

    # 3. 创建 PointCloud 对象
    proj_pcd_p3a_unfilter = o3d.geometry.PointCloud()
    proj_pcd_p3a_unfilter.points = o3d.utility.Vector3dVector(projected_points_p3a)
    # proj_pcd_p3a_unfilter.paint_uniform_color([0, 0, 1])  # 蓝色

    proj_pcd_p3a,ind_p3a = remove_pcd_outlier_dbscan(proj_pcd_p3a_unfilter)
    projected_points_p3a = np.asarray(proj_pcd_p3a.points)

    colors = np.ones((len(proj_pcd_p3a_unfilter.points), 3)) * [1,1,0]
    colors[ind_p3a,:] = [0,0,1]
    proj_pcd_p3a_unfilter.colors = o3d.utility.Vector3dVector(colors)

    #平面中心
    # sphere1 = o3d.geometry.TriangleMesh.create_sphere(radius=0.005)
    # sphere1.paint_uniform_color([1, 1, 0])  
    # sphere1.translate(center_i)

    # sphere2 = o3d.geometry.TriangleMesh.create_sphere(radius=0.005)
    # sphere2.paint_uniform_color([1, 1, 0])  
    # sphere2.translate(center_j)

    # sphere3 = o3d.geometry.TriangleMesh.create_sphere(radius=0.005)
    # sphere3.paint_uniform_color([0, 1, 1]) 
    # sphere3.translate(center_i_outside)

    # sphere4 = o3d.geometry.TriangleMesh.create_sphere(radius=0.005)
    # sphere4.paint_uniform_color([1, 0, 1])  
    # sphere4.translate(center_j_outside)

    # aaaa = o3d.geometry.PointCloud()
    # aaaa.points = o3d.utility.Vector3dVector(np.array(points_between_outside_i+0.0001))
    # aaaa.paint_uniform_color([1, 0, 1])
    # bbbb = o3d.geometry.PointCloud()
    # bbbb.points = o3d.utility.Vector3dVector(np.array(points_between_outside_j+0.0001))
    # bbbb.paint_uniform_color([0, 1, 1])
    # o3d.visualization.draw_geometries([pcd,aaaa,bbbb])
    # 可视化所有内容
    # o3d.visualization.draw_geometries([candidate_TCP_pcd, proj_pcd_outside,sphere1,sphere2,sphere3,sphere4])
    o3d.visualization.draw_geometries([overlap_pcd, proj_pcd_p3a_unfilter.translate([0,0,0.0001])],window_name="Initial TCP & Finger Collision Area (P3a)")

    #*P3b
    center_i_p3b = center_ij + (y_pg/2) * (plane_normals[mmm]) * dist_dir_i
    center_j_p3b = center_ij + (y_pg/2) * (plane_normals[nnn]) * dist_dir_j
    # center_i_p3 = center_ij + (0.02) * (plane_normals[mmm]) * dist_dir_i
    # center_j_p3 = center_ij + (0.02) * (plane_normals[nnn]) * dist_dir_j

    # 1. 筛选出在两个平面之间的点
    points_between_p3b_i,points_beside = select_points_between_planes(points_beside, center_i_p3a, center_i_p3b, plane_normals[mmm])
    points_between_p3b_j,points_beside = select_points_between_planes(points_beside, center_j_p3a, center_j_p3b, plane_normals[nnn])
    points_between_p3b = np.vstack((points_between_p3b_i, points_between_p3b_j))

    # 2. 投影到中间平面
    projected_points_p3b = project_points_to_plane(points_between_p3b, center_ij, plane_normals[mmm])

    # 3. 创建 PointCloud 对象
    proj_pcd_p3b_unfilter = o3d.geometry.PointCloud()
    proj_pcd_p3b_unfilter.points = o3d.utility.Vector3dVector(projected_points_p3b)
    # proj_pcd_p3b_unfilter.paint_uniform_color([0, 0, 1])  # 蓝色

    proj_pcd_p3b,ind_p3b = remove_pcd_outlier_dbscan(proj_pcd_p3b_unfilter)
    projected_points_p3b = np.asarray(proj_pcd_p3b.points)

    colors = np.ones((len(proj_pcd_p3b_unfilter.points), 3)) * [1,1,0]
    colors[ind_p3b,:] = [0,0.5,1]
    proj_pcd_p3b_unfilter.colors = o3d.utility.Vector3dVector(colors)

    o3d.visualization.draw_geometries([overlap_pcd,proj_pcd_p3a_unfilter.translate([0,0,-0.0001]), proj_pcd_p3b_unfilter.translate([0,0,0.0001])],window_name="Initial TCP & Finger Collision Area (P3a+P3b)")

    ##**************************** Plane 4: find beside collision area ****************************

    center_i_p4 = center_i + ((rd + rj)/2) * (plane_normals[mmm]) * dist_dir_i
    center_j_p4 = center_j + ((rd + rj)/2) * (plane_normals[nnn]) * dist_dir_j

    points_between_p4_i,points_beside = select_points_between_planes(points_beside, center_i_p3b, center_i_p4, plane_normals[mmm])
    points_between_p4_j,points_beside = select_points_between_planes(points_beside, center_j_p3b, center_j_p4, plane_normals[nnn])
    points_between_p4 = np.vstack((points_between_p4_i, points_between_p4_j))
    
    projected_points_p4 = project_points_to_plane(points_between_p4, center_ij, plane_normals[mmm])


    # 3. 创建 PointCloud 对象
    proj_pcd_p4_unfilter = o3d.geometry.PointCloud()
    proj_pcd_p4_unfilter.points = o3d.utility.Vector3dVector(projected_points_p4)
    proj_pcd_p4_unfilter.paint_uniform_color([0, 1, 1])

    proj_pcd_p4,ind_p4 = remove_pcd_outlier_dbscan(proj_pcd_p4_unfilter)
    # proj_pcd_p4.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
    projecter_points_p4 = np.asarray(proj_pcd_p4.points)

    colors = np.ones((len(proj_pcd_p4_unfilter.points),3)) * [1,1,0]
    colors[ind_p4,:] = [0,1,1]
    proj_pcd_p4_unfilter.colors = o3d.utility.Vector3dVector(colors)


    o3d.visualization.draw_geometries([overlap_pcd, proj_pcd_p3a.translate([0,0,0.0001]), proj_pcd_p3b.translate([0,0,0.0002]), proj_pcd_p4_unfilter.translate([0,0,-0.0001])],window_name="Initial TCP & Finger Collision Area (P3a+P3b) & Robot Collision Area (P4)")

    #********************** SHOW P1 P2 P3a P3b P4 WITH PCD ****************

    pcd_between_p22 = o3d.geometry.PointCloud()
    pcd_between_p22.points = o3d.utility.Vector3dVector(points_between_p2)
    pcd_between_p22.paint_uniform_color([1, 0, 0])  # 红色

    pcd_between_p33a = o3d.geometry.PointCloud()
    pcd_between_p33a.points = o3d.utility.Vector3dVector(points_between_p3a)
    pcd_between_p33a.paint_uniform_color([0, 0, 1])  # 蓝色

    pcd_between_p33b = o3d.geometry.PointCloud()
    pcd_between_p33b.points = o3d.utility.Vector3dVector(points_between_p3b)
    pcd_between_p33b.paint_uniform_color([0, 0.5, 1])  # 蓝色

    pcd_beside_p4 = o3d.geometry.PointCloud()
    pcd_beside_p4.points = o3d.utility.Vector3dVector(points_beside)
    pcd_beside_p4.paint_uniform_color([0, 1, 1])  # 青色

    o3d.visualization.draw_geometries([pcd_between_p22, pcd_between_p33a.translate([0,0,0.0001]), pcd_between_p33b.translate([0,0,0.0002]), pcd_beside_p4.translate([0,0,-0.0001])],window_name="PCD [P2 P3a P3b P4] in 3D")

    #*Show projected points on P1 P2 P3a P3b P4 with assemble PCD in 3D

    pcd.paint_uniform_color([0.7, 0.7, 0.7])

    o3d.visualization.draw_geometries([pcd, overlap_pcd.translate([0,0,0.0001])],window_name="PCD+P1")
    o3d.visualization.draw_geometries([pcd, proj_pcd_p2.translate([0,0,0.0001])],window_name="PCD+P2")
    o3d.visualization.draw_geometries([pcd, proj_pcd_p3a.translate([0,0,-0.0001])],window_name="PCD+P3a")
    o3d.visualization.draw_geometries([pcd, proj_pcd_p3b.translate([0,0,-0.0001])],window_name="PCD+P3b")
    o3d.visualization.draw_geometries([pcd, proj_pcd_p4.translate([0,0,-0.0001])],window_name="PCD+P4")

    #**************************** P2: Find contours ****************************

    # def auto_img_scale(pcd,
    #                 target_px=1000,          # 想让最长边落在~1000 px
    #                 max_scale=1500,          # 不要比 1500 更大
    #                 min_scale=200):           # 也别太小，避免过密
    #     """
    #     根据 bbox 尺寸自适应 img_scale，并限制在 [min_scale, max_scale] 区间。
    #     """
    #     dists = pcd.compute_nearest_neighbor_distance()   # Δx/Δy 最大值
    #     avg_d  = np.mean(dists)
    #     scale = target_px * 0.0015/avg_d
    #     return max(min(scale, max_scale), min_scale)

    def auto_img_scale(pcd, target_size=512):
        points = np.asarray(pcd.points)
        # 计算原始点云在2D主平面中的宽高
        min_vals = points.min(axis=0)
        max_vals = points.max(axis=0)
        ranges = max_vals - min_vals

        # 选择较大的边作为基准，统一缩放为 target_size
        scale = target_size / np.max(ranges)
        return 1
    #-----------------------chat-cv2------------------------

    def extract_and_visualize_contour_segments_with_normals(pcd, scale=1500, approx_eps_ratio=0.01):
        if isinstance(pcd, o3d.geometry.PointCloud):
            points = np.asarray(pcd.points)
        elif isinstance(pcd, np.ndarray):
            print("Find contours Error: Input is not a PointCloud object.")
            return

        # 1. PCA 主方向（dir1, dir2 构建局部平面）
        pca = PCA(n_components=3)
        pca.fit(points)
        dir1, dir2 = pca.components_[0], pca.components_[1]
        center = pca.mean_

        # 2. 投影到主平面 (2D)
        points = np.dot(points - center, np.vstack([dir1, dir2]).T)


        # 转换点为图像坐标
        img_scale = auto_img_scale(pcd)
        points_img = np.int32((points - points.min(axis=0)) * img_scale)
        img_size = points_img.max(axis=0) + 10

        # 创建空白图像并绘制点
        img = np.zeros((img_size[1], img_size[0]), dtype=np.uint8)
        for pt in points_img:
            cv2.circle(img, tuple(pt), 1, 255, -1)

                # ---- 估计点的典型像素间距（用于选核）----
        # 不依赖第三方：从点图估计一个粗略间距
        # 方案：对网格下采样或用稀疏度近似，这里用形态学距离的近似方法
        ys, xs = np.where(img > 0)
        if len(xs) >= 2:
            # 取一个小窗口统计最近像素距离（简化版估计）
            # 也可换成 scipy.spatial.cKDTree 最近邻求中位数像素间距
            sample = np.random.choice(len(xs), size=min(5000, len(xs)), replace=False)
            pts = np.stack([xs[sample], ys[sample]], axis=1).astype(np.int32)
            # 用小半径腐蚀看能否断开，近似推断间距（保守取值）
            # 简化：用常数兜底
            px_gap = 3
        else:
            px_gap = 3

        # ---- 形态学：先闭运算再开运算 ----
        # 核大小与点间距挂钩，闭运算弥合空隙，开运算去掉毛刺
        k = max(3, int(round(px_gap * 2)))      # 闭运算核（越大越“补全”）
        k_open = max(3, int(round(px_gap * 0.8)))  # 开运算核（轻度去噪）
        kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        kernel_open  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_open, k_open))

        mask = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel_close, iterations=1)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  kernel_open,  iterations=1)

        # ---- 填洞（确保得到真正的外轮廓）----
        h, w = mask.shape
        ff = mask.copy()
        ff = cv2.copyMakeBorder(ff, 1,1,1,1, cv2.BORDER_CONSTANT, value=0)
        cv2.floodFill(ff, None, (0,0), 255)                # 从边界外部泛洪
        ff = ff[1:-1,1:-1]
        holes = cv2.bitwise_not(ff) & cv2.bitwise_not(mask) # 外部区域
        filled = cv2.bitwise_or(mask, cv2.bitwise_not(holes))

        # ---- 去除小连通域（防止孤点影响外轮廓）----
        num, labels, stats, _ = cv2.connectedComponentsWithStats(filled, connectivity=8)
        min_area_px = (k * k) * 2  # 面积阈值：跟核相关
        clean = np.zeros_like(filled)
        for i in range(1, num):
            if stats[i, cv2.CC_STAT_AREA] >= min_area_px:
                clean[labels == i] = 255


        # 使用findContours寻找轮廓
        contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # 绘制各种轮廓处理结果
        img_contours = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        for cnt in contours:
            # 4. 多边形近似（黄色）
            epsilon = 0.01 * cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, epsilon, True)
            cv2.drawContours(img_contours, [approx], 0, (0, 255, 255), 2)

        # 显示轮廓和高级处理结果
        plt.figure(figsize=(10, 6))
        plt.imshow(img_contours)
        plt.title('Contours: Plane 2')
        plt.axis('off')
        plt.show()

        # ---- 将轮廓转换回原始坐标空间 ---- #
        contours_real   = []        # 每个轮廓在 2D 投影坐标系下
        polygons_2d     = []        # shapely Polygon 列表
        # contour_points_list = []
        linesets = []

        for cnt in contours:
            # 近似多边形轮廓
            epsilon = 0.01 * cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, epsilon, True).reshape(-1, 2)

            # 从图像坐标系转换回投影的二维坐标系
            points_2d_back = approx.astype(np.float32) / img_scale + points.min(axis=0)

            # 从二维坐标映射回原始三维空间
            points_3d = np.dot(points_2d_back, np.vstack([dir1, dir2])) + center

            line_segments_2d, line_normals_2d = [], []
            line_segments_3d, line_indices, line_colors = [], [], []

            for i in range(len(points_2d_back)):
                pt1_2d = points_2d_back[i]
                pt2_2d = points_2d_back[(i + 1) % len(points_2d_back)]  # 闭合

                # 线段方向和法线
                vec = pt2_2d - pt1_2d
                length = np.linalg.norm(vec)
                if length == 0:
                    continue
                direction = vec / length
                normal_2d = np.array([-direction[1], direction[0]])

                line_segments_2d.append([pt1_2d, pt2_2d])
                line_normals_2d.append(normal_2d)
                

                # 投影回 3D 空间
                pt1_3d = center + pt1_2d[0]*dir1 + pt1_2d[1]*dir2
                pt2_3d = center + pt2_2d[0]*dir1 + pt2_2d[1]*dir2


                # 线段添加到 LineSet
                idx = len(line_segments_3d)
                line_segments_3d.extend([pt1_3d, pt2_3d])
                line_indices.append([idx, idx + 1])
                color = plt.cm.hsv(i / len(points_2d_back))[:3]
                line_colors.append(color)

            # 构建所有线段
            if len(line_indices) == 0:
                print("没有可视化的线段（可能轮廓太小或被过度简化）")
                return

            line_set = o3d.geometry.LineSet()
            line_set.points = o3d.utility.Vector3dVector(np.asarray(line_segments_3d, dtype=float))
            line_set.lines = o3d.utility.Vector2iVector(np.asarray(line_indices, dtype=np.int32))
            line_set.colors = o3d.utility.Vector3dVector(np.asarray(line_colors, dtype=float))

        # 使用Open3D可视化所有轮廓
        o3d.visualization.draw_geometries([pcd, line_set],window_name="P2 Contour Lines + Normals",width=1280, height=800)

        return line_segments_2d, line_normals_2d, dir1, dir2, center


    contour_segments_2d_p2 = []
    contour_normals_2d_p2 = []
    contour_segments_2d_p2,contour_normals_2d_p2,dir1,dir2,center = extract_and_visualize_contour_segments_with_normals(proj_pcd_p2, scale=1500, approx_eps_ratio=0.01)


    # **************************** Find and Show Initial TCP Box & Test Grid Point ****************************
    def generate_grid_by_spacing(segments_2d, normals_2d, depth=0.05, spacing_edge=0.005,spacing_normal=0.005):
        """
        每条线段沿法线方向扩展构造矩形，并在其中以 spacing 为间距生成等距网格点。
        
        参数：
            segments_2d: List of (pt1, pt2)，二维线段起止点
            normals_2d: List of unit normal vectors，每条线段一个
            depth: 抓取区域宽度（法线方向），单位 m
            spacing: 网格点间隔（单位 m）
            
        返回：
            rectangles: 每个线段对应的矩形（4个点）
            all_grid_points: 每个矩形中生成的点，List[np.ndarray]
        """
        rectangles = []
        all_grid_points = []

        eps=1e-9

        for (pt1, pt2), n in zip(segments_2d, normals_2d):
            pt1 = np.array(pt1)
            pt2 = np.array(pt2)
            n = np.array(n) / np.linalg.norm(n)

            # 线段方向和长度
            dir_vec = pt2 - pt1
            seg_len = np.linalg.norm(dir_vec)
            dir_unit = dir_vec / seg_len

            # 决定方向上的步数
            num_w = int(np.floor((seg_len-eps) / spacing_edge)+1)
            start_spacing_edge = (seg_len-(num_w-1)*spacing_edge)/2.0
            num_d = int(np.floor((depth-eps) / spacing_normal)+1)
            start_spacing_normal = (depth-(num_d-1)*spacing_normal)/2.0      
            if num_w < 1 or num_d < 1:
                continue

            # 构造矩形四个点（逆时针）
            offset = -n * depth
            p1 = pt1 + offset
            p2 = pt2 + offset
            p3 = pt2
            p4 = pt1
            rectangles.append([p1, p2, p3, p4])

            # 在矩形内部生成规则点
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

    def plot_segments_tcpbox_and_grids(segments_2d, rectangles, grid_points):
        """
        在2D坐标平面绘制：
        - 原始线段（蓝色）
        - 每个线段的矩形区域（绿色虚线）
        - 矩形内部的规则网格点（红色 x）
        """
        fig, ax = plt.subplots(figsize=(6, 6))
        used_labels = set()  # 追踪已添加的图例标签

        for (pt1, pt2), rect, grids in zip(segments_2d, rectangles, grid_points):
            # 原始线段
            lbl = 'Edges of Plane2'
            if lbl not in used_labels:
                ax.plot([pt1[0], pt2[0]], [pt1[1], pt2[1]], 'b-', linewidth=3, label=lbl)
                used_labels.add(lbl)
            else:
                ax.plot([pt1[0], pt2[0]], [pt1[1], pt2[1]], 'b-', linewidth=3)

            # 矩形区域（闭合）
            rect = np.array(rect + [rect[0]])
            lbl = 'Initial TCP Box'
            if lbl not in used_labels:
                ax.plot(rect[:, 0], rect[:, 1], 'g--', linewidth=0.5, label=lbl)
                used_labels.add(lbl)
            else:
                ax.plot(rect[:, 0], rect[:, 1], 'g--', linewidth=0.5)

            # 网格点
            grids = np.array(grids)
            lbl = 'Test Grid Points'
            if lbl not in used_labels:
                ax.plot(grids[:, 0], grids[:, 1], 'rx', markersize=4, label=lbl)
                used_labels.add(lbl)
            else:
                ax.plot(grids[:, 0], grids[:, 1], 'rx', markersize=4)

        ax.set_aspect('equal')
        ax.set_title("2D: Edges, TCP Box, and Test Grid Points")
        ax.grid(True)
        ax.legend()
        plt.tight_layout()
        plt.show()


    tcp_box,test_grid_points = generate_grid_by_spacing(contour_segments_2d_p2, contour_normals_2d_p2, depth=b_pg+c_pg, spacing_edge=z_pg/5, spacing_normal=b_pg/5)
    plot_segments_tcpbox_and_grids(contour_segments_2d_p2,tcp_box,test_grid_points)

    # Show each TCP Boxes and it's test grid points
    def highlight_segment_rect_grid(segments_2d, rectangles, grid_points):
        """
        始终显示所有线段，只高亮当前索引对应的矩形与网格点。
        """
        all_pts = np.array(list(itertools.chain.from_iterable(segments_2d)))
        min_xy = all_pts.min(axis=0) - 0.025
        max_xy = all_pts.max(axis=0) + 0.025


        for i in range(len(segments_2d)):
            fig, ax = plt.subplots(figsize=(6, 6))
            ax.set_title(f"TCP Box and Test Grid Points for Edges {i+1}/{len(segments_2d)}")

            # 所有线段：蓝色
            lbl = 'Edges of Plane2'
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
                    start_point_line = mid - normal_clockwise_90 * 0.018
                    end_point_line = start_point_line + normal_clockwise_90 * 0.015
                    end_point_base1 = end_point_line + vec_12 * 0.005 - normal_clockwise_90 * 0.005
                    end_point_base2 = end_point_line - vec_12 * 0.005 + normal_clockwise_90 * 0.005
                    end_point_finger1 = end_point_base1 + normal_clockwise_90 * 0.008
                    end_point_finger2 = end_point_base2 + normal_clockwise_90 * 0.008

                    ax.plot([start_point_line[0], end_point_line[0]], [start_point_line[1], end_point_line[1]], 'm', linewidth=1.5,label='Gripper Direction')
                    ax.plot([end_point_base1[0], end_point_base2[0]], [end_point_base1[1], end_point_base2[1]], 'm', linewidth=1.5)
                    ax.plot([end_point_base1[0], end_point_finger1[0]], [end_point_base1[1], end_point_finger1[1]], 'm', linewidth=1.5)
                    ax.plot([end_point_base2[0], end_point_finger2[0]], [end_point_base2[1], end_point_finger2[1]], 'm', linewidth=1.5)
                    
                if lbl not in used_labels:
                    ax.plot([pt1[0], pt2[0]], [pt1[1], pt2[1]], 'b-', linewidth=3,label=lbl)
                    used_labels.add(lbl)
                else:
                    ax.plot([pt1[0], pt2[0]], [pt1[1], pt2[1]], 'b-', linewidth=3)




            # 当前的矩形：绿色
            rect = np.array(rectangles[i] + [rectangles[i][0]])
            ax.plot(rect[:, 0], rect[:, 1], 'g--', linewidth=0.5, label='Initial TCP Box')

            # 当前的 grid 点：红色
            grids = np.array(grid_points[i])
            if grid_points is not None:
                ax.plot(grids[:, 0], grids[:, 1], 'rx', markersize=4, label='Test Grid Points')

            ax.set_xlim(min_xy[0], max_xy[0])
            ax.set_ylim(min_xy[1], max_xy[1])
            ax.set_aspect('equal')
            ax.grid(True)
            ax.legend()
            plt.tight_layout()
            plt.show()


    # highlight_segment_rect_grid(contour_segments_2d_p2, tcp_box, test_grid_points)


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


    def show_gripper_bounding_box(segments_2d, tcp_box, shapes):
        all_pts = [pt for seg in segments_2d for pt in seg]
        bounds = np.array(all_pts)
        min_xy = bounds.min(axis=0) - 0.01
        max_xy = bounds.max(axis=0) + 0.01

        for i, segment_shape in enumerate(shapes):
            for j, shape in enumerate(segment_shape):
                fig, ax = plt.subplots(figsize=(8, 8))
                ax.set_title(f"Edge {i+1}, Point {j+1}:Bounding Boxes of Gripper and Robot Arm")

                used_labels = set()

                # 所有线段（蓝）
                lbl = 'Edges of Plane2'
                for pt1, pt2 in segments_2d:
                    if lbl not in used_labels:
                        ax.plot([pt1[0], pt2[0]], [pt1[1], pt2[1]], 'b-', linewidth=1, label=lbl)
                        used_labels.add(lbl)
                    else:
                        ax.plot([pt1[0], pt2[0]], [pt1[1], pt2[1]], 'b-', linewidth=1)

                # 初始矩形
                rect = np.array(tcp_box[i] + [tcp_box[i][0]])
                lbl = 'Initial TCP Box'
                if lbl not in used_labels:
                    ax.plot(rect[:, 0], rect[:, 1], 'g--', linewidth=0.5, label=lbl)
                    used_labels.add(lbl)
                else:
                    ax.plot(rect[:, 0], rect[:, 1], 'g--', linewidth=0.5)

                # 当前测试点
                pt = shape['point']
                lbl = 'Test Point'
                if lbl not in used_labels:
                    ax.plot(pt[0], pt[1], 'ro', markersize=4, label=lbl)
                    used_labels.add(lbl)
                else:
                    ax.plot(pt[0], pt[1], 'ro', markersize=4)

                #6个矩形
                colors = ['red', 'purple', 'orange','deepskyblue','yellow','limegreen']
                Box_label = ['Finger Safe Space Box', 'Finger Box', 'Finger Base Box', 'Robot Arm Box','Gripper Area Box','Robot Back Space Box']
                for k, rect in enumerate(shape['rectangles']):
                    poly = np.array(rect + [rect[0]])
                    lbl = Box_label[k]
                    if lbl not in used_labels:
                        ax.plot(poly[:, 0], poly[:, 1], color=colors[k], linewidth=1.5, label=lbl)
                        used_labels.add(lbl)
                    else:
                        ax.plot(poly[:, 0], poly[:, 1], color=colors[k], linewidth=1.5)

                # 梯形
                # trap = np.array(shape['trapezoid'] + [shape['trapezoid'][0]])
                # lbl = 'Robot Arm'
                # if lbl not in used_labels:
                #     ax.plot(trap[:, 0], trap[:, 1], color='deepskyblue', linestyle='--', linewidth=1.2, label=lbl)
                #     used_labels.add(lbl)
                # else:
                #     ax.plot(trap[:, 0], trap[:, 1], color='deepskyblue', linestyle='--', linewidth=1.2)

                ax.set_xlim(min_xy[0], max_xy[0])
                ax.set_ylim(min_xy[1], max_xy[1])
                ax.set_aspect('equal')
                ax.grid(True)
                ax.legend()
                plt.tight_layout()
                plt.show()

    points_and_gripper_bounding_box = create_gripper_bounding_box(test_grid_points, contour_segments_2d_p2)
    # show_gripper_bounding_box(contour_segments_2d_p2,tcp_box,points_and_gripper_bounding_box)



    #************************ Project P134 to 2 (CV2) **********************


    #*******************************





    def get_plane_contour_polygon(pcd,dir1,dir2,center):

        if pcd.is_empty():
            return Polygon()

        points = np.asarray(pcd.points)

        if points.size <= 50:
            return Polygon()

        # 2. 投影到主平面 (2D)
        points = np.dot(points - center, np.vstack([dir1, dir2]).T)


        # 转换点为图像坐标
        img_scale = auto_img_scale(pcd)
        points_img = np.int32((points - points.min(axis=0)) * img_scale)
        img_size = points_img.max(axis=0) + 10

        # 创建空白图像并绘制点
        img = np.zeros((img_size[1], img_size[0]), dtype=np.uint8)
        for pt in points_img:
            cv2.circle(img, tuple(pt), 1, 255, -1)

                # ---- 估计点的典型像素间距（用于选核）----
        # 不依赖第三方：从点图估计一个粗略间距
        # 方案：对网格下采样或用稀疏度近似，这里用形态学距离的近似方法
        ys, xs = np.where(img > 0)
        if len(xs) >= 2:
            # 取一个小窗口统计最近像素距离（简化版估计）
            # 也可换成 scipy.spatial.cKDTree 最近邻求中位数像素间距
            sample = np.random.choice(len(xs), size=min(5000, len(xs)), replace=False)
            pts = np.stack([xs[sample], ys[sample]], axis=1).astype(np.int32)
            # 用小半径腐蚀看能否断开，近似推断间距（保守取值）
            # 简化：用常数兜底
            px_gap = 3
        else:
            px_gap = 3

        # ---- 形态学：先闭运算再开运算 ----
        # 核大小与点间距挂钩，闭运算弥合空隙，开运算去掉毛刺
        k = max(3, int(round(px_gap * 2)))      # 闭运算核（越大越“补全”）
        k_open = max(3, int(round(px_gap * 0.8)))  # 开运算核（轻度去噪）
        kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        kernel_open  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_open, k_open))

        mask = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel_close, iterations=1)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  kernel_open,  iterations=1)

        # ---- 填洞（确保得到真正的外轮廓）----
        h, w = mask.shape
        ff = mask.copy()
        ff = cv2.copyMakeBorder(ff, 1,1,1,1, cv2.BORDER_CONSTANT, value=0)
        cv2.floodFill(ff, None, (0,0), 255)                # 从边界外部泛洪
        ff = ff[1:-1,1:-1]
        holes = cv2.bitwise_not(ff) & cv2.bitwise_not(mask) # 外部区域
        filled = cv2.bitwise_or(mask, cv2.bitwise_not(holes))

        # ---- 去除小连通域（防止孤点影响外轮廓）----
        num, labels, stats, _ = cv2.connectedComponentsWithStats(filled, connectivity=8)
        min_area_px = (k * k) * 2  # 面积阈值：跟核相关
        clean = np.zeros_like(filled)
        for i in range(1, num):
            if stats[i, cv2.CC_STAT_AREA] >= min_area_px:
                clean[labels == i] = 255


        # 使用findContours寻找轮廓
        contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # 绘制各种轮廓处理结果
        img_contours = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        for cnt in contours:
            # 4. 多边形近似（黄色）
            epsilon = 0.01 * cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, epsilon, True)
            cv2.drawContours(img_contours, [approx], 0, (0, 255, 255), 2)

        # 显示轮廓和高级处理结果
        plt.figure(figsize=(10, 6))
        plt.imshow(img_contours)
        plt.title('Contour Dectection Result')
        plt.axis('off')
        plt.show()




        # ---- 将轮廓转换回原始坐标空间 ---- #
        contours_real   = []        # 每个轮廓在 2D 投影坐标系下
        polygons_2d     = []        # shapely Polygon 列表
        # contour_points_list = []
        linesets = []

        for cnt in contours:
            # 近似多边形轮廓
            epsilon = 0.01 * cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, epsilon, True).reshape(-1, 2)

            # 从图像坐标系转换回投影的二维坐标系
            points_2d_back = approx.astype(np.float32) / img_scale + points.min(axis=0)

            # 从二维坐标映射回原始三维空间
            points_3d = np.dot(points_2d_back, np.vstack([dir1, dir2])) + center

            contours_real.append(points_2d_back)
            if points_2d_back.size <= 4:
                polygons_2d.append(Polygon())
            else:
                polygons_2d.append(Polygon(points_2d_back))
            # contour_points_list.append(points_3d)

            # 构造Open3D LineSet对象
            num_points = points_3d.shape[0]
            lines = [[i, (i+1)%num_points] for i in range(num_points)]

            colors = [[1, 0, 0] for _ in lines]  # 红色线条

            line_set = o3d.geometry.LineSet()
            line_set.points = o3d.utility.Vector3dVector(points_3d)
            line_set.lines = o3d.utility.Vector2iVector(lines)
            line_set.colors = o3d.utility.Vector3dVector(colors)

            linesets.append(line_set)

        # 原始点云设置颜色便于观察
        pcd.paint_uniform_color([0.5, 0.5, 0.5])

        # 使用Open3D可视化所有轮廓
        o3d.visualization.draw_geometries([pcd, *linesets], window_name='Plane Contours 3D View')





        # # 轮廓转回真实二维坐标
        # # contour_real = approx.reshape(-1, 2).astype(np.float32)/img_scale + points.min(axis=0)
        # # poly_contour = Polygon(contour_real)

        # # 判断点是否在轮廓内
        # test_point = (0.0, 0.02)
        # # is_inside = polygons_2d.contains(Point(test_point))
        # # print(f"点{test_point}是否位于轮廓内：{is_inside}")
        # is_inside = any(poly.contains(Point(test_point)) for poly in polygons_2d)
        # print(f"点 {test_point} 是否位于某个轮廓内：{is_inside}")

        # # 计算相交面积（实际单位）
        # intersection_areas = []
        # rect_real = np.array([[-0.05,-0.01], [0.05,-0.01], [0.05,-0.05], [-0.05,-0.05]])
        # poly_rect = Polygon(rect_real)

        # for idx, poly in enumerate(polygons_2d):
        #     intersection = poly.intersection(poly_rect)
        #     area = intersection.area
        #     intersection_areas.append(area)
        # print(f"最大相交面积：{max(intersection_areas)*1e4:.2f} cm^2")
        # # intersection_area = polygons_2d.intersection(poly_rect).area
        # # print(f"实际相交面积：{intersection_area*1e4}(cm^2)")


        # # ---开始绘图---
        # plt.figure(figsize=(10, 8))

        # # 1. 绘制原始点云
        # plt.scatter(points[:, 0], points[:, 1], s=2, color='blue', alpha=0.5, label='Pointcloud')

        # # 2. 绘制轮廓（红色）
        # used_labels = set()
        # lbl = 'Contour'
        # for contour_real in contours_real:
        #     if lbl not in used_labels:
        #         contour_plot = np.vstack([contour_real, contour_real[0]])  # 闭合轮廓
        #         plt.plot(contour_plot[:,0], contour_plot[:,1], color='red', linewidth=2,label=lbl)
        #         used_labels.add(lbl)
        #     else:
        #         contour_plot = np.vstack([contour_real, contour_real[0]])  # 闭合轮廓
        #         plt.plot(contour_plot[:,0], contour_plot[:,1], color='red', linewidth=2)

        # # 3. 绘制测试点
        # point_color = 'green' if is_inside else 'red'
        # plt.scatter(*test_point, color=point_color, s=100, marker='*', label='Test TCP Point')

        # # 4. 绘制矩形
        # rect_plot = np.vstack([rect_real, rect_real[0]])  # 闭合矩形
        # plt.plot(rect_plot[:,0], rect_plot[:,1], color='purple', linewidth=2, linestyle='--', label='Test Gripper Bounding Box')

        # # 额外标注面积结果
        # centroid = poly_rect.centroid.coords[0]
        # plt.text(centroid[0], centroid[1], f'Area={(max(intersection_areas)*1e4):.2f}', color='purple', fontsize=12, ha='center')

        # # 设置其他显示选项
        # plt.title('Pointcloud, Contour, Test TCP Point, and Test Gripper Bounding Box')
        # plt.xlabel('X')
        # plt.ylabel('Y')
        # plt.legend(loc='best')
        # plt.axis('equal')
        # plt.grid(True)

        # plt.show()

        return polygons_2d

    plane_contour_polygon_list = [
        get_plane_contour_polygon(overlap_pcd,dir1,dir2,center),
        get_plane_contour_polygon(proj_pcd_p2,dir1,dir2,center),
        get_plane_contour_polygon(proj_pcd_p3a,dir1,dir2,center),
        get_plane_contour_polygon(proj_pcd_p3b,dir1,dir2,center),
        get_plane_contour_polygon(proj_pcd_p4,dir1,dir2,center),
        ]

    # plane_contour_polygon_list = [polygon_p1,polygon_p2,polygon_p3,polygon_p4]

    #********************** Loop Find feasible TCP *********************
    from shapely.validation import explain_validity, make_valid   # Shapely>=2.0
    from shapely import set_precision
    from shapely.geometry import Polygon, Point
    from shapely.errors import GEOSException
    from shapely.ops import unary_union
    GRID_SIZE = 1e-9

    def _clean_geom(geom, name=""):
        if geom.is_empty:
            return geom

        g = geom

        # 1) 修复无效
        if not g.is_valid:
            g = make_valid(g)
        if not g.is_valid:
            g = g.buffer(0)

        # 2) set_precision：放到后面更稳妥
        try:
            g = set_precision(g, GRID_SIZE)
        except Exception as e:
            print(f"[WARN] set_precision failed on '{name}': {e}")

        # 3) 合并
        try:
            if hasattr(g, "geoms"):
                g = unary_union(g)
        except Exception:
            pass

        # 4) 最终有效性检查
        if not g.is_valid:
            msg = explain_validity(g)
            print(f"[WARN] Geometry '{name}' still invalid after cleaning: {msg}")

        return g

    def _safe_intersection_area(a, b):
        """
        在交集时启用 grid_size（snap rounding），并在异常时回退到 buffer(0)
        """
        try:
            # Shapely 2.x 的交集允许传 grid_size
            return a.intersection(b, grid_size=GRID_SIZE).area
        except GEOSException:
            a2 = _clean_geom(a, "A@fallback")
            b2 = _clean_geom(b, "B@fallback")
            return a2.intersection(b2, grid_size=GRID_SIZE).area
        

    def find_feasible_tcp(plane_contour_polygon_list,all_shapes):

        filtered_shapes = []
        feasible_points_on_edge = []
        intersection_areas_on_edge =[]
        min_area = 0.15 * (z_pg-2*rj) * (b_pg-2*rj)

        # 先把 plane_contour_polygon_list 里的 Polygon 统一成 list，并清洗
        for i in range(5):
            lst = plane_contour_polygon_list[i]
            if isinstance(lst, Polygon):
                plane_contour_polygon_list[i] = [lst]
            # 对每个多边形做清洗与设精度
            plane_contour_polygon_list[i] = [_clean_geom(p, f"plane_poly_{i}") for p in plane_contour_polygon_list[i]]

        poly_p1_list = plane_contour_polygon_list[0]
        poly_p2_list = plane_contour_polygon_list[1]
        poly_p3a_list = plane_contour_polygon_list[2]
        poly_p3b_list = plane_contour_polygon_list[3]        
        poly_p4_list = plane_contour_polygon_list[4]


        for segment_shapes in all_shapes:
            filtered_segment = []
            feasible_point = []
            intersection_areas = []
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
                    for poly in poly_p3a_list 
                )
                condition_4 = all(
                    not poly.intersects(rect3_geom) and
                    not poly.intersects(rect4_geom)
                    for poly in poly_p3b_list
                )
                condition_5 = all(not poly.intersects(rect4_geom) for poly in poly_p4_list)

                if  condition_1 and condition_2 and condition_3 and condition_4 and condition_5:
                    filtered_segment.append(shape)
                    feasible_point.append(pt)
                    intersection_areas.append(total_intersection_areas)
                # if condition_1 :
                #     filtered_segment.append(shape)
                #     point_1.append(pt)                

            filtered_shapes.append(filtered_segment)
            feasible_points_on_edge.append(feasible_point)
            intersection_areas_on_edge.append(intersection_areas)
        return filtered_shapes,feasible_points_on_edge,intersection_areas_on_edge
    
#********************* Original Version *************
    # def find_feasible_tcp(plane_contour_polygon_list,all_shapes):

    #     filtered_shapes = []
    #     feasible_points_on_edge = []
    #     intersection_areas_on_edge =[]
    #     min_area = 0.3 * z_pg * b_pg

    #     for segment_shapes in all_shapes:
    #         filtered_segment = []
    #         feasible_point = []
    #         intersection_areas = []
    #         for shape in segment_shapes:
    #             pt = shape['point']
    #             rectangles = shape['rectangles']
                
    #             point_geom = Point(pt)
    #             rect1_geom = Polygon(rectangles[0])  # Finger tip Safe Space
    #             rect2_geom = Polygon(rectangles[1])  # Finger length
    #             rect3_geom = Polygon(rectangles[2])  # Gripper Base
    #             rect4_geom = Polygon(rectangles[3])  # Robot arm 
    #             rect5_geom = Polygon(rectangles[4])  # Gripper Area
    #             rect6_geom = Polygon(rectangles[5])  # Robot back sapace Box

    #             for i in range(4):
    #                 lst = plane_contour_polygon_list[i]
    #                 if isinstance(lst, Polygon):
    #                     plane_contour_polygon_list[i] = [lst]

    #             poly_0_list = plane_contour_polygon_list[0]
    #             poly_1_list = plane_contour_polygon_list[1]
    #             poly_2_list = plane_contour_polygon_list[2]
    #             poly_3_list = plane_contour_polygon_list[3]

    #             total_intersection_areas = sum(poly.intersection(rect5_geom).area for poly in poly_0_list)

    #             # condition_1 = any(poly.contains(point_geom) for poly in poly_0_list)
    #             condition_2 = total_intersection_areas > min_area 
    #             condition_3 = all(
    #                 not poly.intersects(rect3_geom) and not poly.intersects(rect4_geom)
    #                 for poly in poly_1_list
    #             )
    #             condition_4 = all(
    #                 not poly.intersects(rect1_geom) and
    #                 not poly.intersects(rect2_geom) and
    #                 not poly.intersects(rect3_geom) and
    #                 not poly.intersects(rect4_geom)
    #                 for poly in poly_2_list
    #             )
    #             condition_5 = all(not poly.intersects(rect4_geom) for poly in poly_3_list)

    #             if  condition_2 and condition_3 and condition_4 and condition_5:
    #                 filtered_segment.append(shape)
    #                 feasible_point.append(pt)
    #                 intersection_areas.append(total_intersection_areas)
    #             # if condition_1 :
    #             #     filtered_segment.append(shape)
    #             #     point_1.append(pt)                

    #         filtered_shapes.append(filtered_segment)
    #         feasible_points_on_edge.append(feasible_point)
    #         intersection_areas_on_edge.append(intersection_areas)
    #     return filtered_shapes,feasible_points_on_edge,intersection_areas_on_edge


    feasible_TCP_and_shapes,feasible_TCP,intersection_areas = find_feasible_tcp(plane_contour_polygon_list,points_and_gripper_bounding_box)
    # feasible_TCP = [shape['point'] for segment in feasible_TCP_and_shapes for shape in segment]

    # highlight_segment_rect_grid(contour_segments_2d_p2,tcp_box,feasible_TCP)


    #*********************** Ranking function ******************************************

    def get_area_score(intersection_areas):
        area_scores = []
        max_area = max((z_pg - 2*rj) * (b_pg - 2*rj), 1e-9)  # avoid 0
        for areas in intersection_areas:
            arr = np.asarray(areas, dtype=float)
            s = (arr - 0.15*max_area) / (0.85*max_area)
            s = np.clip(s, 0.0, 1.0)  # a<0.15*max -> 0；a>max -> 1
            area_scores.append(s)
        return area_scores


    # def get_center_score(TCP_points, center_pcd, dir1, dir2, center):
    #     """
    #     For each segment i, returns a list of center-based scores (one per feasible TCP point).
    #     Closer to the projected center => higher score (normalized to [0,1]).
    #     """
    #     center_pcd = np.asarray(center_pcd, dtype=float)
    #     center = np.asarray(center, dtype=float)
    #     # project the 3D center into the same 2D basis as TCP_points
    #     basis = np.vstack([dir1, dir2]).T  # shape (3,2) or (2,2) depending on your setup
    #     center_local = np.dot(center_pcd - center, basis)

    #     center_scores = []
    #     eps = 1e-12
    #     for pts in TCP_points:
    #         if not pts:                 # no feasible points on this segment
    #             center_scores.append([])
    #             continue
    #         pts_np = np.asarray(pts, dtype=float).reshape(-1, 2)
    #         d = np.linalg.norm(pts_np - center_local, axis=1)
    #         if d.max() - d.min() < eps:
    #             scores = np.ones_like(d)      # all same distance -> give all 1.0
    #         else:
    #             scores = 1.0 - (d - d.min()) / (d.max() - d.min())
    #         center_scores.append(scores.tolist())
    #     return center_scores

    def get_center_score(TCP_points, center_pcd, dir1, dir2, center):

        center_pcd = np.asarray(center_pcd, dtype=float)
        center = np.asarray(center, dtype=float)
        dir1 = np.asarray(dir1, dtype=float)
        dir2 = np.asarray(dir2, dtype=float)

        basis = np.vstack([dir1, dir2])
        
        TCP_points_dist = []
        
        for pts in TCP_points:
            if not pts:
                TCP_points_dist.append(np.array([],dtype=float))
                continue
            uv = np.asarray(pts, dtype=float).reshape(-1, 2)
            p_3d = center + uv @ basis
            dist = np.linalg.norm(p_3d - center_pcd,axis=1)
            TCP_points_dist.append(dist)


        non_empty = [d for d in TCP_points_dist if d.size > 0]
        if len(non_empty) == 0:
            return [d.copy() for d in TCP_points_dist]
        max_dist = np.max([np.max(d) for d in non_empty])

        if max_dist == 0:
        # all points are the same at center of mass -> give all 1.0
            TCP_dist_scores = [np.ones_like(d) for d in TCP_points_dist]
        else:
            TCP_dist_scores = [(1.0 - d/max_dist) for d in TCP_points_dist]

        return TCP_dist_scores




    def rank_feasible_tcp(feasible_TCP, intersection_areas):
        w1, w2 = 0.1, 0.9
        # 2D mean of original_points
        means = np.mean(original_points, axis=0)

        area_scores = get_area_score(intersection_areas)                      # list[list[float]]
        center_scores = get_center_score(feasible_TCP, means, dir1, dir2, center)  # list[list[float]]

        ranked = []
        for c_seg, a_seg in zip(center_scores, area_scores):
            # ensure same length per segment (should match by construction)
            m = min(len(c_seg), len(a_seg))
            ranked.append([w1 * c_seg[k] + w2 * a_seg[k] for k in range(m)])
        return ranked
    feasible_TCP_rank = rank_feasible_tcp(feasible_TCP,intersection_areas)





    def highlight_feasible_tcp(TCP_points,TCP_rank, segments_2d, tcp_box):
        """
        遍历 filtered_shapes 中的每组 shape，依次高亮展示：
        - 所有线段（蓝色）
        - 当前点的 TCP 矩形（绿色）
        - 当前点坐标（红色）

        参数：
        - filtered_shapes: List[List[dict]]，每个 shape 包含 point, rectangles
        - segments_2d: List of 2D line segments [(pt1, pt2), ...]
        - rectangles_per_shape: 同 filtered_shapes 结构，用于提供 TCP box（rectangles[0]）
        """

        all_pts = np.array(list(itertools.chain.from_iterable(segments_2d)))
        min_xy = all_pts.min(axis=0) - 0.02
        max_xy = all_pts.max(axis=0) + 0.02


        for i,pt in enumerate(TCP_points):
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.set_title(f"Edge {i+1}: Feasible TCP and TCP Box")

            # 所有线段：蓝色
            used_labels = set()
            lbl = 'Coutours of Plane 2'
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
                    start_point_line = mid - normal_clockwise_90 * 0.018
                    end_point_line = start_point_line + normal_clockwise_90 * 0.015
                    end_point_base1 = end_point_line + vec_12 * 0.005 - normal_clockwise_90 * 0.005
                    end_point_base2 = end_point_line - vec_12 * 0.005 + normal_clockwise_90 * 0.005
                    end_point_finger1 = end_point_base1 + normal_clockwise_90 * 0.008
                    end_point_finger2 = end_point_base2 + normal_clockwise_90 * 0.008

                    ax.plot([start_point_line[0], end_point_line[0]], [start_point_line[1], end_point_line[1]], 'm', linewidth=1.5,label='Gripper Direction')
                    ax.plot([end_point_base1[0], end_point_base2[0]], [end_point_base1[1], end_point_base2[1]], 'm', linewidth=1.5)
                    ax.plot([end_point_base1[0], end_point_finger1[0]], [end_point_base1[1], end_point_finger1[1]], 'm', linewidth=1.5)
                    ax.plot([end_point_base2[0], end_point_finger2[0]], [end_point_base2[1], end_point_finger2[1]], 'm', linewidth=1.5)
                    
                if lbl not in used_labels:
                    ax.plot([pt1[0], pt2[0]], [pt1[1], pt2[1]], 'b-', linewidth=1, label=lbl)
                    used_labels.add(lbl)
                else:
                    ax.plot([pt1[0], pt2[0]], [pt1[1], pt2[1]], 'b-', linewidth=1)

            # 当前点：绿色点
            pt = np.array(pt).reshape(-1, 2)
            # print(pt.shape)          # 看维度
            # print(pt.ndim)
            if pt.size:
                scores_i = np.array(TCP_rank[i], dtype=float)  # (Ni,)
                if scores_i.shape[0] != pt.shape[0]:
                    print(f"[warn] edge {i}: #scores({scores_i.shape[0]}) != #pts({pt.shape[0]})")
                    # 可选：截断或跳过
                    m = min(scores_i.shape[0], pt.shape[0])
                    pt = pt[:m]
                    scores_i = scores_i[:m]
                # ax.plot(pt[:,0], pt[:,1],linestyle='None', marker='x', color='lime', label='Feasible TCP Point')
                scatter = plt.scatter(pt[:, 0], pt[:, 1], c=scores_i, cmap='RdYlGn',vmin=0, vmax=1, s=10, label='Feasible TCP Point')
                cbar = plt.colorbar(scatter, ax=ax, label='Score', fraction=0.046, pad=0.04)
                cbar.set_ticks([0.0, 0.5, 1.0])  # 最小、中间、最大
                cbar.set_ticklabels([f"0.0", f"0.5", f"1.0"])
            else:
                print("No feasible TCP point found!")
                ax.plot([], [],linestyle='None', marker='x', color='lime', label='Feasible TCP Point')
                # scatter = plt.scatter([], [], c=scores_i, cmap='RdYlGn', s=100, label='Feasible TCP Point')   

            # 当前矩形框（rectangles[0]）：绿色虚线框
            rect = np.array(tcp_box[i] + [tcp_box[i][0]])  # 闭合多边形
            ax.plot(rect[:, 0], rect[:, 1], 'g--', linewidth=2, label='TCP Box')


            ax.set_xlim(min_xy[0], max_xy[0])
            ax.set_ylim(min_xy[1], max_xy[1])
            ax.set_aspect('equal')
            ax.grid(True)
            ax.legend()
            plt.tight_layout()
            plt.show()

    highlight_feasible_tcp(feasible_TCP,feasible_TCP_rank,contour_segments_2d_p2,tcp_box)  # 实际上不需要额外传，因为内部已包含 rectangles



    def highlight_feasible_all_tcp(TCP_points, TCP_rank, segments_2d, tcp_box):
        """
        展示：
        - 所有线段（蓝色）
        - 各边对应的可行 TCP 点（按分数着色）
        - 各边对应的 TCP 矩形框（绿色虚线）

        参数：
        - TCP_points: List[np.ndarray (Ni,2)]，每条边的 TCP 点集
        - TCP_rank:   List[np.ndarray (Ni,)]，与 TCP_points 对应的分数
        - segments_2d: List[ (pt1, pt2) ]，每条边的两个端点 (2,)
        - tcp_box:    List[np.ndarray (4,2)] 或 (M,2)，每条边的矩形四点（按顺序）
        """

        # 收集所有点用于设定坐标范围
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
        pad = 0.02
        min_xy = all_xy.min(axis=0) - pad
        max_xy = all_xy.max(axis=0) + pad

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.set_title("All Feasible TCP and TCP Box")

        used_labels = set()
        scatter_handle = None
        cbar = None

        # 同步遍历四者，避免索引错位
        for (pt1, pt2), tcp, rect, scores in zip(segments_2d, TCP_points, tcp_box, TCP_rank):
            # 1) 线段-蓝色
            lbl = 'Contours on Plane'
            if lbl not in used_labels:
                ax.plot([pt1[0], pt2[0]], [pt1[1], pt2[1]], 'b-', linewidth=3, label=lbl)
                used_labels.add(lbl)
            else:
                ax.plot([pt1[0], pt2[0]], [pt1[1], pt2[1]], 'b-', linewidth=3)

            # 2) TCP 点-按分数着色
            tcp = np.asarray(tcp) if tcp is not None else np.empty((0,2))
            scores = np.asarray(scores, dtype=float) if scores is not None else np.empty((0,))
            if tcp.ndim == 1 and tcp.size == 2:
                tcp = tcp.reshape(1, 2)

            if tcp.size > 0:
                # 对齐长度
                m = min(len(tcp), len(scores))
                if m == 0:
                    pass
                else:
                    tcp = tcp[:m]
                    scores = scores[:m]
                    # 尝试根据数据范围设置色条范围
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
                        cbar = plt.colorbar(scatter_handle, ax=ax, label='Score', fraction=0.046, pad=0.04)

            # 3) TCP 矩形-绿色虚线（闭合）
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
        ax.legend(loc='best')
        plt.tight_layout()
        plt.show()

    highlight_feasible_all_tcp(feasible_TCP,feasible_TCP_rank,contour_segments_2d_p2,tcp_box)