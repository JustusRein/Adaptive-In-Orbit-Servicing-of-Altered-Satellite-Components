import json
import math
import copy
import numpy as np
import open3d as o3d
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
import os 


# ---------------------------
# 数据结构
# ---------------------------

@dataclass
class RegistrationParams:
    """注册与检测的全局参数"""
    voxel_size: Optional[float] = None  # 若为 None，将根据装配体尺寸自适应
    # 预处理
    normal_radius_mult: float = 2.0
    fpfh_radius_mult: float = 5.0
    # 匹配距离阈值（与体素大小关联）
    ransac_dist_mult: float = 1.5
    icp_dist_mult: float = 1.5
    # FGR 与 RANSAC 开关顺序（先快后稳）
    use_fgr_first: bool = True
    # 缺损检测
    missing_dist_mult: float = 2.0  # full->defect 最近邻距离超过该阈值则判定为缺失
    # 缺损聚类（DBSCAN）
    dbscan_eps_mult: float = 2.5
    dbscan_min_points: int = 30
    # 去噪（可选）
    use_statistical_outlier_removal: bool = True
    sor_nb_neighbors: int = 30
    sor_std_ratio: float = 2.0


@dataclass
class PartResult:
    name: str
    transform_to_assembly: np.ndarray
    pcd_in_assembly: o3d.geometry.PointCloud
    num_points: int
    coarse_fitness: float  # <-- 新增


@dataclass
class DefectCluster:
    cluster_id: int
    size: int
    center: Tuple[float, float, float]
    extent: Tuple[float, float, float]  # AABB 尺寸


@dataclass
class DefectReport:
    part_name: str
    missing_points: int
    total_points: int
    missing_ratio: float
    clusters: List[DefectCluster]


# ---------------------------
# 工具函数
# ---------------------------

def load_pcd(path: str) -> o3d.geometry.PointCloud:
    p = o3d.io.read_point_cloud(path)
    if p.is_empty():
        raise ValueError(f"Point cloud is empty: {path}")
    return p


def voxel_down_and_est_normals(pcd: o3d.geometry.PointCloud, voxel: float,
                               normal_radius: float) -> o3d.geometry.PointCloud:
    p = pcd.voxel_down_sample(voxel)
    p.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(
            radius=normal_radius, max_nn=30
        )
    )
    p.normalize_normals()
    return p


def compute_fpfh(pcd: o3d.geometry.PointCloud, radius: float) -> o3d.pipelines.registration.Feature:
    return o3d.pipelines.registration.compute_fpfh_feature(
        pcd,
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius, max_nn=100)
    )


def auto_voxel_size_from_bbox(pcd: o3d.geometry.PointCloud, divisor: int = 200) -> float:
    bbox = pcd.get_axis_aligned_bounding_box()
    diag = np.linalg.norm(bbox.get_extent())
    # 对于工业件，diag/200 是较稳的起点；你可以按数据密度调整
    return max(diag / divisor, 1e-6)


def maybe_denoise(pcd: o3d.geometry.PointCloud, params: RegistrationParams) -> o3d.geometry.PointCloud:
    if not params.use_statistical_outlier_removal:
        return pcd
    p, _ = pcd.remove_statistical_outlier(nb_neighbors=params.sor_nb_neighbors,
                                          std_ratio=params.sor_std_ratio)
    return p


def preprocess_for_registration(pcd: o3d.geometry.PointCloud, voxel: float,
                                params: RegistrationParams):
    normal_radius = params.normal_radius_mult * voxel
    fpfh_radius = params.fpfh_radius_mult * voxel
    p_down = voxel_down_and_est_normals(pcd, voxel, normal_radius)
    fpfh = compute_fpfh(p_down, fpfh_radius)
    return p_down, fpfh


def try_fgr(source_down, target_down, source_fpfh, target_fpfh, max_corr_dist):
    option = o3d.pipelines.registration.FastGlobalRegistrationOption(
        maximum_correspondence_distance=max_corr_dist
    )
    result = o3d.pipelines.registration.registration_fgr_based_on_feature_matching(
        source_down, target_down, source_fpfh, target_fpfh, option
    )
    return result


def try_ransac(source_down, target_down, source_fpfh, target_fpfh, max_corr_dist):
    distance_threshold = max_corr_dist
    checkers = [
        o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
        o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(distance_threshold)
    ]
    result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        source_down, target_down, source_fpfh, target_fpfh,
        mutual_filter=True,
        max_correspondence_distance=distance_threshold,
        estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
        ransac_n=4,
        checkers=checkers,
        criteria=o3d.pipelines.registration.RANSACConvergenceCriteria(400000, 500)
    )
    return result


def refine_icp_point_to_plane(source, target, init_T, icp_dist):
    # Use a robust kernel if available to resist outliers (e.g., missing areas)
    try:
        estimation = o3d.pipelines.registration.TransformationEstimationPointToPlane(
            kernel=o3d.pipelines.registration.TukeyLoss(k=icp_dist)
        )
    except TypeError:
        # Older Open3D: robust kernels not supported in this constructor
        estimation = o3d.pipelines.registration.TransformationEstimationPointToPlane()

    # Ensure normals exist (point-to-plane needs them)
    if not source.has_normals():
        source.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=icp_dist, max_nn=30)
        )
        source.normalize_normals()
    if not target.has_normals():
        target.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=icp_dist, max_nn=30)
        )
        target.normalize_normals()

    result = o3d.pipelines.registration.registration_icp(
        source, target, icp_dist, init_T,
        estimation_method=estimation,
        criteria=o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=100)
    )
    return result


def compose_point_cloud(parts: List[o3d.geometry.PointCloud]) -> o3d.geometry.PointCloud:
    """将多个点云拼接为一个新的点云（复制顶点）"""
    all_pts = []
    all_cols = []
    for p in parts:
        all_pts.append(np.asarray(p.points))
        if p.has_colors():
            all_cols.append(np.asarray(p.colors))
        else:
            # 若无颜色，填默认灰色
            all_cols.append(np.full((np.asarray(p.points).shape[0], 3), 0.5))
    pts = np.vstack(all_pts)
    cols = np.vstack(all_cols)
    out = o3d.geometry.PointCloud()
    out.points = o3d.utility.Vector3dVector(pts)
    out.colors = o3d.utility.Vector3dVector(cols)
    return out


def random_color(seed: int) -> Tuple[float, float, float]:
    rng = np.random.default_rng(seed)
    return tuple(rng.random(3).tolist())

def crop_by_aabb(pcd: o3d.geometry.PointCloud, aabb: o3d.geometry.AxisAlignedBoundingBox):
    if aabb is None:
        return pcd
    return pcd.crop(aabb)


# ---------------------------
# 1) 零件 -> 装配体：逐件配准
# ---------------------------

def register_part_to_assembly(part: o3d.geometry.PointCloud,
                              assembly: o3d.geometry.PointCloud,
                              params: RegistrationParams,
                              name: str = "part",
                              target_aabbs: Optional[List[o3d.geometry.AxisAlignedBoundingBox]] = None
                              ) -> PartResult:
    # ...（保留原始去噪与参数准备）

    # Helper: 单次（在一个给定 ROI 上）做粗-精配准并评分
    def _one_roi_register(asm_roi: o3d.geometry.PointCloud):
        voxel = params.voxel_size or auto_voxel_size_from_bbox(asm_roi)
        part_down, part_fpfh = preprocess_for_registration(part, voxel, params)
        asm_down, asm_fpfh = preprocess_for_registration(asm_roi, voxel, params)
        max_corr = params.ransac_dist_mult * voxel

        # 粗配准候选
        cands = []
        if params.use_fgr_first:
            cands.append(try_fgr(part_down, asm_down, part_fpfh, asm_fpfh, max_corr))
            # 当“Too few correspondences”时，试一个不 mutual_filter 的 RANSAC 回退
            try:
                cands.append(try_ransac(part_down, asm_down, part_fpfh, asm_fpfh, max_corr))
            except Exception:
                pass
        else:
            try:
                cands.append(try_ransac(part_down, asm_down, part_fpfh, asm_fpfh, max_corr))
            except Exception:
                pass
            cands.append(try_fgr(part_down, asm_down, part_fpfh, asm_fpfh, max_corr))

        best_init = max(cands, key=lambda r: r.fitness) if cands else None

        # 若粗配准失败，给个单位阵并让评分极差
        if best_init is None or best_init.fitness <= 1e-6:
            return None, -1.0, float("inf"), np.eye(4)

        # 精配准：点到面；不稳定再回退点到点
        icp_dist = params.icp_dist_mult * voxel
        try:
            icp_res = refine_icp_point_to_plane(part, asm_roi, best_init.transformation, icp_dist)
        except Exception:
            icp_res = o3d.pipelines.registration.registration_icp(
                part, asm_roi, icp_dist, best_init.transformation,
                estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(),
                criteria=o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=80)
            )

        # 评分：优先看 fitness（内点比例），同时参考 inlier_rmse（越小越好）
        score_fitness = float(icp_res.fitness)
        score_rmse = float(icp_res.inlier_rmse)
        return icp_res, score_fitness, score_rmse, best_init.transformation

    # 若没有 ROI，等价于整个装配体作为唯一 ROI
    rois = target_aabbs if target_aabbs else [None]
    best = None
    for ridx, aabb in enumerate(rois):
        asm_roi = crop_by_aabb(assembly, aabb) if aabb is not None else assembly
        if asm_roi.is_empty():
            continue
        res, fit, rmse, _ = _one_roi_register(asm_roi)
        if res is None:
            continue
        # 选“fitness 最大，若相等则 rmse 最小”的方案
        key = (fit, -rmse)
        if (best is None) or (key > best[0]):
            best = (key, res, ridx)

    if best is None:
        # 彻底失败的兜底：直接返回不变换版本
        p_fail = copy.deepcopy(part)
        p_fail.paint_uniform_color([1, 0, 0])  # 红色表明失败
        return PartResult(name=name, transform_to_assembly=np.eye(4),
                          pcd_in_assembly=p_fail, num_points=np.asarray(part.points).shape[0],
                          coarse_fitness=0.0)

    _, icp_res, chosen_ridx = best
    part_in_asm = copy.deepcopy(part)
    part_in_asm.transform(icp_res.transformation)
    col = random_color(abs(hash(name)) % (2**32))
    part_in_asm.paint_uniform_color(col)

    return PartResult(
        name=name,
        transform_to_assembly=icp_res.transformation,
        pcd_in_assembly=part_in_asm,
        num_points=np.asarray(part.points).shape[0],
        coarse_fitness=float(icp_res.fitness)
    )

def register_parts_to_assembly(parts: Dict[str, o3d.geometry.PointCloud],
                               assembly: o3d.geometry.PointCloud,
                               params: RegistrationParams,
                               part_rois: Optional[Dict[str, List[o3d.geometry.AxisAlignedBoundingBox]]] = None
                               ) -> Tuple[List[PartResult], o3d.geometry.PointCloud]:
    results: List[PartResult] = []
    used_roi_ids = set()  # (part_name, roi_index) 或者全局编号，这里用全局编号更简单
    
    for name, p in parts.items():
        print(f"[INFO] Registering part -> assembly: {name}")
        rois = part_rois.get(name) if part_rois else None
        res = register_part_to_assembly(p, assembly, params, name=name, target_aabbs=rois)
        print(f"       coarse_fitness={res.coarse_fitness:.3f}")
        results.append(res)

                # === 新增：实时可视化 ===
        current_part_clouds = [r.pcd_in_assembly for r in results]
        vis_geoms = [assembly] + current_part_clouds  # 显示原始装配体 + 当前所有已配准零件
        print(f"[VISUALIZE] Showing assembly + {len(results)} part(s)")
        o3d.visualization.draw_geometries(vis_geoms,
            window_name=f"Assembly + {len(results)} parts",
            width=1280, height=720)

    # 拼装由零件组成的装配点云
    parts_in_asm = [r.pcd_in_assembly for r in results]
    assembly_from_parts = compose_point_cloud(parts_in_asm)
    return results, assembly_from_parts


# ---------------------------
# 3) “零件组成的装配点云” <-> 缺损装配点云：配准 & 缺损检测
# ---------------------------

def align_full_to_defect(full_asm: o3d.geometry.PointCloud,
                         defect: o3d.geometry.PointCloud,
                         params: RegistrationParams) -> Tuple[o3d.geometry.PointCloud, np.ndarray]:
    """将'由零件组成的装配点云'配准到'缺损装配点云'坐标"""
    full_asm = maybe_denoise(full_asm, params)
    defect = maybe_denoise(defect, params)

    voxel = params.voxel_size or auto_voxel_size_from_bbox(full_asm)
    full_down, full_fpfh = preprocess_for_registration(full_asm, voxel, params)
    def_down, def_fpfh = preprocess_for_registration(defect, voxel, params)

    max_corr = params.ransac_dist_mult * voxel
    # 同样尝试 FGR/RANSAC 并精配准
    cands = []
    if params.use_fgr_first:
        cands.append(try_fgr(full_down, def_down, full_fpfh, def_fpfh, max_corr))
        cands.append(try_ransac(full_down, def_down, full_fpfh, def_fpfh, max_corr))
    else:
        cands.append(try_ransac(full_down, def_down, full_fpfh, def_fpfh, max_corr))
        cands.append(try_fgr(full_down, def_down, full_fpfh, def_fpfh, max_corr))
    best_init = max(cands, key=lambda r: r.fitness)

    icp_dist = params.icp_dist_mult * voxel
    icp_res = refine_icp_point_to_plane(full_asm, defect, best_init.transformation, icp_dist)

    full_in_def = copy.deepcopy(full_asm)
    full_in_def.transform(icp_res.transformation)
    return full_in_def, icp_res.transformation


def compute_missing_mask(full_in_def: o3d.geometry.PointCloud,
                         defect: o3d.geometry.PointCloud,
                         params: RegistrationParams) -> np.ndarray:
    """
    对齐后（full_in_def 与 defect 在同一坐标系），
    计算 full->defect 最近邻距离，超过阈值则视为缺失。
    """
    voxel = params.voxel_size or auto_voxel_size_from_bbox(full_in_def)
    missing_thresh = params.missing_dist_mult * voxel

    # Open3D 自带的最近邻距离（每个 full 点到 defect 的最近邻）
    dists = np.asarray(full_in_def.compute_point_cloud_distance(defect))
    missing_mask = dists > missing_thresh
    return missing_mask


def cluster_missing_points(missing_pcd: o3d.geometry.PointCloud,
                           params: RegistrationParams) -> List[DefectCluster]:
    if missing_pcd.is_empty():
        return []

    voxel = params.voxel_size or auto_voxel_size_from_bbox(missing_pcd)
    eps = params.dbscan_eps_mult * voxel
    labels = np.array(missing_pcd.cluster_dbscan(
        eps=eps, min_points=params.dbscan_min_points, print_progress=False
    ))
    clusters = []
    for cid in sorted(set(labels)):
        if cid == -1:
            # -1 为噪声；如需也输出可放开
            continue
        idx = np.where(labels == cid)[0]
        sub = missing_pcd.select_by_index(idx)
        aabb = sub.get_axis_aligned_bounding_box()
        center = tuple(aabb.get_center().tolist())
        extent = tuple(aabb.get_extent().tolist())
        clusters.append(DefectCluster(cluster_id=cid, size=len(idx),
                                      center=center, extent=extent))
    return clusters


def analyze_missing_by_part(full_in_def: o3d.geometry.PointCloud,
                            defect: o3d.geometry.PointCloud,
                            part_point_ranges: List[Tuple[str, int, int]],
                            params: RegistrationParams) -> Tuple[List[DefectReport],
                                                                 o3d.geometry.PointCloud]:
    """
    根据全装配（由零件拼成）在缺损坐标系下的点顺序切分，统计每个零件的缺失。
    part_point_ranges: List[(part_name, start_idx, end_idx)]  指明每个零件在 full 点云中的连续索引范围。
    返回：
      - 每个零件的缺损统计（含聚类）
      - 用颜色可视化的'缺损高亮'点云（绿色=存在；红色=缺失）
    """
    missing_mask = compute_missing_mask(full_in_def, defect, params)
    pts = np.asarray(full_in_def.points)

    # 彩色可视化：存在=绿，缺失=红
    vis = copy.deepcopy(full_in_def)
    colors = np.zeros_like(pts)
    colors[~missing_mask] = np.array([0.1, 0.8, 0.1])  # 存在：绿色
    colors[missing_mask] = np.array([0.9, 0.1, 0.1])   # 缺失：红色
    vis.colors = o3d.utility.Vector3dVector(colors)

    reports: List[DefectReport] = []
    for (name, s, e) in part_point_ranges:
        rng_mask = np.zeros_like(missing_mask, dtype=bool)
        rng_mask[s:e] = True
        part_missing = missing_mask & rng_mask
        missing_idx = np.where(part_missing)[0]
        total = e - s
        ratio = (missing_idx.size / max(total, 1))

        # 聚类该零件的缺失
        if missing_idx.size > 0:
            part_missing_pcd = full_in_def.select_by_index(missing_idx)
            clusters = cluster_missing_points(part_missing_pcd, params)
        else:
            clusters = []

        reports.append(DefectReport(
            part_name=name,
            missing_points=int(missing_idx.size),
            total_points=int(total),
            missing_ratio=float(ratio),
            clusters=clusters
        ))

    return reports, vis


# ---------------------------
# 便捷封装：一键执行完整流程
# ---------------------------

def run_pipeline(assembly_path: str,
                 part_paths: Dict[str, str],
                 defect_path: str,
                 params: Optional[RegistrationParams] = None,
                 save_dir: Optional[str] = None):
    """
    assembly_path: 装配体点云（完整）
    part_paths:    dict: {零件名: 点云路径}
    defect_path:   缺损装配体点云
    params:        RegistrationParams
    save_dir:      可选，保存中间/结果文件的目录
    """
    params = params or RegistrationParams()

    # 读取
    assembly = load_pcd(assembly_path)
    parts = {name: load_pcd(p) for name, p in part_paths.items()}
    defect = load_pcd(defect_path)

        # Step 1 & 2: 零件 -> 装配体；拼装由零件组成的装配点云
    def aabb_from_minmax(xmin, ymin, zmin, xmax, ymax, zmax):
        return o3d.geometry.AxisAlignedBoundingBox(min_bound=[xmin, ymin, zmin],
                                                max_bound=[xmax, ymax, zmax])

    part_rois = {
        "part_B": [aabb_from_minmax(0,0,0, 340,340,340)],
        "part_C": [aabb_from_minmax(0,0,340, 340,50,340)],
        "part_D": [aabb_from_minmax(9,0,0, 340,340,340)],
        # 其他零件可不设（=全局装配为搜索域）
    }
    # 然后调用：
    part_results, assembly_from_parts = register_parts_to_assembly(parts, assembly, params, part_rois=part_rois)

    # 为后续“缺损归属统计”记录每个零件在拼装大点云中的索引范围
    ranges: List[Tuple[str, int, int]] = []
    cur = 0
    ordered_parts_in_asm: List[o3d.geometry.PointCloud] = []
    for pr in part_results:
        n = np.asarray(pr.pcd_in_assembly.points).shape[0]
        ranges.append((pr.name, cur, cur + n))
        cur += n
        ordered_parts_in_asm.append(pr.pcd_in_assembly)
    full_asm = compose_point_cloud(ordered_parts_in_asm)

    # Step 3: full_asm（由零件组成的装配点云） -> 缺损点云：配准
    full_in_def, T_full_to_def = align_full_to_defect(full_asm, defect, params)

    # 按零件统计缺损并生成可视化点云
    reports, vis_colored = analyze_missing_by_part(full_in_def, defect, ranges, params)

    # 可选保存
    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)  # <-- 确保 out/ 存在
        o3d.io.write_point_cloud(f"{save_dir}/assembly_from_parts.pcd", assembly_from_parts)
        o3d.io.write_point_cloud(f"{save_dir}/full_in_defect.pcd", full_in_def)
        o3d.io.write_point_cloud(f"{save_dir}/defect_colored_missing.pcd", vis_colored)
        with open(f"{save_dir}/transforms.json", "w") as f:
            json.dump({
                "full_to_defect": T_full_to_def.tolist(),
                "parts_to_assembly": {
                    pr.name: pr.transform_to_assembly.tolist() for pr in part_results
                }
            }, f, indent=2, ensure_ascii=False)
    with open(f"{save_dir}/defect_report.json", "w", encoding="utf-8") as f:
        json.dump([{
            "part_name": r.part_name,
            "missing_points": int(r.missing_points),
            "total_points": int(r.total_points),
            "missing_ratio": float(r.missing_ratio),
            "clusters": [{
                "cluster_id": int(c.cluster_id),
                "size": int(c.size),
                "center": [float(x) for x in c.center],
                "extent": [float(x) for x in c.extent]
            } for c in r.clusters]
        } for r in reports], f, indent=2, ensure_ascii=False)

    # 返回关键信息
    return {
        "parts_to_assembly": {pr.name: pr.transform_to_assembly for pr in part_results},
        "full_to_defect": T_full_to_def,
        "defect_reports": reports,
        "colored_missing_pcd": vis_colored
    }


# ---------------------------
# 使用示例（请按你的数据路径修改）
# ---------------------------
if __name__ == "__main__":
    # 路径示例（请替换成你的实际文件）
    assembly_path = r"D:\Codecouldcode\099.MA_Hanyu\Object\Verification_examples\00_CubeSat_Assembly_sampled.pcd"           # 完整装配体点云
    part_paths = {
        "part_A": r"D:\Codecouldcode\099.MA_Hanyu\Object\Verification_examples\01_Bottom_CubeSat_sampled.pcd",
        "part_B": r"D:\Codecouldcode\099.MA_Hanyu\Object\Verification_examples\02_Wall_no_Logo_sampled.pcd",
        "part_C": r"D:\Codecouldcode\099.MA_Hanyu\Object\Verification_examples\02_Wall_with_acor_logo_sampled.pcd",
        "part_D": r"D:\Codecouldcode\099.MA_Hanyu\Object\Verification_examples\02_wall_with_PLCM_logo_plcm_sampled.pcd",
        "part_E": r"D:\Codecouldcode\099.MA_Hanyu\Object\Verification_examples\03_Board_Empty_sampled.pcd",
    }
    defect_path = r"D:\Codecouldcode\099.MA_Hanyu\Object\Verification_examples\01_Bottom_CubeSat_sampled.pcd"           # 缺损装配点云

    params = RegistrationParams(
        voxel_size=None,            # 让程序按装配尺度自适应；若数据很密可手动设大些
        normal_radius_mult=2.5,
        fpfh_radius_mult=7.0,
        ransac_dist_mult=2.0,
        icp_dist_mult=2.0,
        use_fgr_first=True,
        missing_dist_mult=2.2,
        dbscan_eps_mult=3.0,
        dbscan_min_points=40,
        use_statistical_outlier_removal=True,
        sor_nb_neighbors=30,
        sor_std_ratio=2.0
    )

    results = run_pipeline(assembly_path, part_paths, defect_path, params, save_dir="out")

    # 打印简单摘要
    print("\n=== 缺损统计（按零件） ===")
    for r in results["defect_reports"]:
        print(f"- {r.part_name}: missing {r.missing_points}/{r.total_points} "
              f"({r.missing_ratio*100:.2f}%)  clusters={len(r.clusters)}")

    # 如需立即可视化（Open3D 窗口）
    o3d.visualization.draw_geometries([results["colored_missing_pcd"]])
