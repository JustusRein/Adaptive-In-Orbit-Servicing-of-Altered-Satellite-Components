import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt

# ---------- 文件路径 ----------
cad_path = r"D:\Codecouldcode\099.MA_Hanyu\00_project\nontextured.ply"          # 你的CAD模型（STL格式）
scan_path = r"D:\Codecouldcode\099.MA_Hanyu\00_project\merged_cloud.ply"          # 实际测得点云（.ply 或 .pcd）

# ---------- Step 1: 读取并采样 CAD 模型为点云 ----------
mesh = o3d.io.read_triangle_mesh(cad_path)
mesh.compute_vertex_normals()
cad_pcd = mesh.sample_points_uniformly(number_of_points=30000)

# ---------- Step 2: 读取扫描点云 ----------
scan_pcd = o3d.io.read_point_cloud(scan_path)

# ---------- Step 3: 法线估计 ----------
cad_pcd.estimate_normals()
scan_pcd.estimate_normals()

# ---------- Step 4: 粗略对齐（可选，如果坐标系相差很远） ----------
# 可设置初始变换矩阵（比如单位矩阵）
init_transform = np.identity(4)

# ---------- Step 5: ICP 精细配准 ----------
threshold = 0.01  # 最大点对距离（单位：m）
reg = o3d.pipelines.registration.registration_icp(
    scan_pcd, cad_pcd, threshold, init_transform,
    o3d.pipelines.registration.TransformationEstimationPointToPlane())

# ---------- Step 6: 应用变换 ----------
scan_pcd.transform(reg.transformation)

# ---------- Step 7: 计算误差 ----------
distances = cad_pcd.compute_point_cloud_distance(scan_pcd)
distances = np.asarray(distances)
mean_error = np.mean(distances)
max_error = np.max(distances)

print("Is the Point Cloud in CAD Database?：")
print(f"Mean Error: {mean_error*1000:.2f} mm")
print(f"Max Error: {max_error*1000:.2f} mm")

# ---------- Step 8: 可视化 ----------
# 给 scan_pcd 添加误差着色
colors = plt.cm.jet(distances / np.max(distances))[:, :3]
scan_pcd.colors = o3d.utility.Vector3dVector(colors)

o3d.visualization.draw_geometries([cad_pcd.paint_uniform_color([0.5,0.5,0.5]), scan_pcd],
                                   window_name='CAD vs Scanned Point Cloud')

