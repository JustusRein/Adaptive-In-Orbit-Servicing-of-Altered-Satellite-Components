import numpy as np
import open3d as o3d
from grasp_store import GraspStore, GraspData
import matplotlib.pyplot as plt
import logging, pathlib
import copy
from pathlib import Path
import yaml


#****************************** Set all necessary paths and TARGET before execution. *****************************

pcd_path = Path("Journal/Pointlcloud clustered Vulkan subassembly/2141302 - TRAVA DA ALAVANCA FEHD-II - PART 1.f3d.pcd")

npz_load_path = Path("results/20260627_014646-Franka-2141302 - TRAVA DA ALAVANCA FEHD-II - PART 1.f3d-0.1/grasp_data/20260627_014646-Franka-2141302 - TRAVA DA ALAVANCA FEHD-II - PART 1.f3d-0.1.npz")

npy_load_path = Path("results/20260627_014646-Franka-2141302 - TRAVA DA ALAVANCA FEHD-II - PART 1.f3d-0.1/grasp_data/candidate.npy")

gripper_path = Path("01_project/gripper_parameter/Franka.yaml")

# the target to be visualized
target_pair = 3
target_edge = 2
target_point = 8
TARGET = 'Best'  # 'Best' or 'Worst' or 'Select'

#******************************************************************************************************************


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




# parameters for 3d view
size_axis = 30.0
size_sphere = 3.0

loaded = GraspStore.load_npz(npz_load_path)

if loaded.best() == None:
    print("=============== There is no Feasible Grasp Found ===============")
    exit()

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
    
pcd = o3d.io.read_point_cloud(pcd_path)
pcd.scale(pcd_scaler(pcd), pcd.get_center())
pcd.paint_uniform_color([0.6, 0.6, 0.6])

# *=================== Start =======================================
if TARGET == 'Best':
    target = loaded.best()
elif TARGET == 'Worst':
    target = loaded.worst()
elif TARGET == 'Select':
    target = loaded.get_by_index(pair_num=target_pair,edge_num=target_edge,point_num=target_point)
target_pair = target.pair_num
target_edge = target.edge_num




#Search：The i-th plane pair of the k-th pair
subset_pair = loaded.filter_by_indices(pair_num=target_pair)
points_pair = np.array([r.TCP for r in subset_pair], dtype=float)    # shape (N, 3)
scores_pair = np.array([r.score for r in subset_pair], dtype=float)

#Search：The i-th plane pair on the j-th edge of the k-th pair
subset_edge = loaded.filter_by_indices(pair_num=target_pair, edge_num=target_edge)
points_edge = np.array([r.TCP for r in subset_edge], dtype=float)    # shape (N, 3)
scores_edge = np.array([r.score for r in subset_edge], dtype=float)

colors_pair = plt.cm.RdYlGn(np.asarray(scores_pair, dtype=float))[:, :3]
pcd_tcp_alledge = o3d.geometry.PointCloud()
pcd_tcp_alledge.points = o3d.utility.Vector3dVector(points_pair)
pcd_tcp_alledge.colors = o3d.utility.Vector3dVector(colors_pair)

colors_edge = plt.cm.RdYlGn(np.asarray(scores_edge, dtype=float))[:, :3]
pcd_tcp_oneedge = o3d.geometry.PointCloud()
pcd_tcp_oneedge.points = o3d.utility.Vector3dVector(points_edge)
pcd_tcp_oneedge.colors = o3d.utility.Vector3dVector(colors_edge)



# # Target point
Ra, Rb, Rc = target.pose  # 3 direction vectors
R_rows = np.vstack([Ra, Rb, Rc])
print("TCP (world):", target.TCP)
print("Pose (rows):\n", R_rows)

T = GraspStore.grasp_to_transform(target) # generate transformation matrix

U, _, Vt = np.linalg.svd(T[:3, :3])
T[:3, :3] = U @ Vt

sphere_targetTCP = o3d.geometry.TriangleMesh.create_sphere(radius=size_sphere)
sphere_targetTCP.compute_vertex_normals()
sphere_targetTCP.paint_uniform_color([1.0, 0.0, 1.0])  # purple
sphere_targetTCP.translate(np.asarray(target.TCP, dtype=float))

frame_targetTCP = o3d.geometry.TriangleMesh.create_coordinate_frame(size=size_axis, origin=[0, 0, 0])
frame_targetTCP.transform(T)

frame_world = o3d.geometry.TriangleMesh.create_coordinate_frame(size=size_axis, origin=[0, 0, 0])




# o3d.visualization.draw_geometries([pcd_tcp_alledge, pcd,sphere_targetTCP,frame_targetTCP,frame_world],window_name='Feasible TCP [all edge] in 3D')


# o3d.visualization.draw_geometries([pcd_tcp_oneedge, pcd,sphere_targetTCP,frame_targetTCP,frame_world],window_name='Feasible TCP [one edge] in 3D')




#===========================    construct grippermesh     ======================================

def create_box_centered(dx, dy, dz, color):
    """
    create a box centered at origin and return an Open3D TriangleMesh
    dx, dy, dz: (mm)
    """
    box = o3d.geometry.TriangleMesh.create_box(width=dx, height=dy, depth=dz)
    box.translate(np.array([-dx/2, -dy/2, -dz/2], dtype=float))
    box.compute_vertex_normals()
    box.paint_uniform_color(color)
    return box

def T_from_translation(t):
    """4x4 transform metris for translation only"""
    Tt = np.eye(4, dtype=float)
    Tt[:3, 3] = np.asarray(t, dtype=float)
    return Tt

def apply_world_T(geom, Tworld):
    """apply transformation to geometry in world coordinate system"""
    geom.transform(Tworld)
    return geom



grasp_width = target.grasp_width if target.grasp_width is not None else 40.0 

# dimension definition of gripper mesh

#Finger 
finger_len_x = b_pg + c_pg + rj 
finger_wid_z = a_pg + v_pg + w_pg  
finger_dep_y = e_pg + 2*(i_pg + rj)
finger_color = [0.5, 0.0, 0.5]
left_finger_local_center  = np.array([-finger_len_x/2, 0.0, - (grasp_width/2 + finger_wid_z/2 + 0.5)], dtype=float) # move 1 mm in z-axis for showing the grasp contact area
right_finger_local_center  = np.array([-finger_len_x/2, 0.0, + (grasp_width/2 + finger_wid_z/2 + 0.5)], dtype=float) # move 1 mm in z-axis for showing the grasp contact area

#Fingertip clearance
fingertip_len_x = x_pg + rj
fingertip_wid_z = a_pg + v_pg + w_pg 
fingertip_dep_y = e_pg + 2*(i_pg + rj)
fingertip_color = [1.0, 0.0, 0.0]   
left_fingertip_local_center  = np.array([fingertip_len_x/2, 0.0, - (grasp_width/2 + fingertip_wid_z/2 + 0.5)], dtype=float)
right_fingertip_local_center  = np.array([fingertip_len_x/2, 0.0, + (grasp_width/2 + fingertip_wid_z/2 + 0.5)], dtype=float)

#Gripper base
base_len_x = d_pg + t_pg + u_pg + rj
base_wid_z = h_pg +2*k_pg
base_dep_y = j_pg + 2*rj
base_color = [1.0, 0.647,0.0] 
base_local_center  = np.array([-(finger_len_x+base_len_x/2), 0.0, 0.0,], dtype=float)

#Robot arm
arm_len_x = rc + rf + 2*rj
arm_wid_z = q_pg + 2*r_pg
arm_dep_y = rd + re + 2*rj
arm_color = [0.0, 0.749,1.0] 
arm_local_center  = np.array([-(finger_len_x+base_len_x+arm_len_x/2), 0.0, 0.0,], dtype=float)

#Robot Back Space
back_len_x = x_pg + rj
back_wid_z = q_pg + 2*r_pg
back_dep_y = rd + re + 2*rj
back_color = [0.196, 0.804,0.196] 
back_local_center  = np.array([-(finger_len_x+base_len_x+arm_len_x+back_len_x/2), 0.0, 0.0,], dtype=float)

#Gripper contact area
area_len_x = b_pg - 2*rj
area_wid_z = 0.5  # 0.5 mm for showing the grasp contact area
area_dep_y = z_pg - 2*rj
area_color = [1.0, 1.0, 0.0]
left_area_local_center  = np.array([-area_len_x/2, 0.0, - (grasp_width/2 + area_wid_z/2)], dtype=float)
right_area_local_center  = np.array([-area_len_x/2, 0.0, + (grasp_width/2 + area_wid_z/2)], dtype=float)

# ========= constract gripper in local coordinate system =========

left_finger  = create_box_centered(finger_len_x, finger_dep_y, finger_wid_z, finger_color)
right_finger = create_box_centered(finger_len_x, finger_dep_y, finger_wid_z, finger_color)
left_fingertip = create_box_centered(fingertip_len_x,  fingertip_dep_y,  fingertip_wid_z,  fingertip_color)
right_fingertip = create_box_centered(fingertip_len_x,  fingertip_dep_y,  fingertip_wid_z,  fingertip_color)
base = create_box_centered(base_len_x,  base_dep_y,  base_wid_z,  base_color)
arm = create_box_centered(arm_len_x,  arm_dep_y,  arm_wid_z,  arm_color)
arm_back = create_box_centered(back_len_x,  back_dep_y,  back_wid_z,  back_color)
left_area = create_box_centered(area_len_x,  area_dep_y,  area_wid_z,  area_color)
right_area = create_box_centered(area_len_x,  area_dep_y,  area_wid_z,  area_color)

# trasnlate gripper to right position
left_finger.transform(T_from_translation(left_finger_local_center))
right_finger.transform(T_from_translation(right_finger_local_center))
left_fingertip.transform(T_from_translation(left_fingertip_local_center))
right_fingertip.transform(T_from_translation(right_fingertip_local_center))
base.transform(T_from_translation(base_local_center))
arm.transform(T_from_translation(arm_local_center))
arm_back.transform(T_from_translation(back_local_center))
left_area.transform(T_from_translation(left_area_local_center))
right_area.transform(T_from_translation(right_area_local_center))



# ========= Transform gripper to world coordinate system =========

left_finger_world  = copy.deepcopy(left_finger)
right_finger_world = copy.deepcopy(right_finger)
left_fingertip_world  = copy.deepcopy(left_fingertip)
right_fingertip_world  = copy.deepcopy(right_fingertip)
base_world  = copy.deepcopy(base)
arm_world  = copy.deepcopy(arm)
back_world  = copy.deepcopy(arm_back)
left_area_world  = copy.deepcopy(left_area)
right_area_world  = copy.deepcopy(right_area)


apply_world_T(left_finger_world,  T)
apply_world_T(right_finger_world, T)
apply_world_T(left_fingertip_world,  T)
apply_world_T(right_fingertip_world,  T)
apply_world_T(base_world, T)
apply_world_T(arm_world,  T)
apply_world_T(back_world,  T)
apply_world_T(left_area_world,  T)
apply_world_T(right_area_world,  T)

gripper_mesh = [left_finger_world,right_finger_world,left_fingertip_world,right_fingertip_world,base_world,arm_world,back_world,left_area_world,right_area_world]
#===========================    End gripper construct    ======================================
o3d.visualization.draw_geometries([pcd],window_name='PCD',mesh_show_back_face=True)
o3d.visualization.draw_geometries([pcd_tcp_alledge, pcd,sphere_targetTCP,frame_targetTCP]+gripper_mesh,window_name='Feasible TCP (single pair)['+TARGET+' case]: in 3D',mesh_show_back_face=True)
o3d.visualization.draw_geometries([pcd,sphere_targetTCP,frame_targetTCP]+gripper_mesh,window_name='Feasible TCP (single pair)['+TARGET+' case]: in 3D',mesh_show_back_face=True)

# '''
# ===================== visualization =====================

SPHERE_R_MM   = 0.8      
CROP_MARGIN   = 20.0   

def sphere_at(xyz, r, color):
    m = o3d.geometry.TriangleMesh.create_sphere(radius=r)
    m.compute_vertex_normals()
    m.paint_uniform_color(color)
    m.translate(np.asarray(xyz, dtype=float))
    return m



# 20260512: Visualize pcd and all candidate TCP
# ===================== visualization : ALL Candidate TCP (Sphere) =====================

candidate_TCP_3d = np.load(npy_load_path, allow_pickle=True)

candidate_spheres = []

all_candidate_points = []

for pair_group in candidate_TCP_3d:
    for edge in pair_group:
        for p in edge:

            # s = sphere_at(
            #     xyz=np.asarray(p, dtype=float),
            #     r=SPHERE_R_MM,
            #     color=[0.0, 0.0, 1.0]   # blue
            # )
            # candidate_spheres.append(s)
            all_candidate_points.append(np.asarray(p, dtype=float))
all_candidate_points = np.array(all_candidate_points)
pcd_candidate = o3d.geometry.PointCloud()
pcd_candidate.points = o3d.utility.Vector3dVector(all_candidate_points)    
candidate_colors = np.tile(np.array([[0.0, 0.0, 1.0]]), (len(all_candidate_points), 1))
pcd_candidate.colors = o3d.utility.Vector3dVector(candidate_colors)       

# o3d.visualization.draw_geometries(
#     [pcd] + candidate_spheres,
#     window_name='ALL Candidate TCPs'
# )

o3d.visualization.draw_geometries(
    [pcd, pcd_candidate],
    window_name='ALL Candidate TCPs'
)

#20260512: Visualize all feasible TCP and PCD

# ===================== visualization : ALL Feasible TCP (Sphere) =====================

all_feasible = loaded.all()

all_feasible_scores = np.array(
    [g.score for g in all_feasible],
    dtype=float
)

colors_all_feasible = plt.cm.RdYlGn(all_feasible_scores)[:, :3]

feasible_spheres = []

for g, col in zip(all_feasible, colors_all_feasible):

    s = sphere_at(
        xyz=np.asarray(g.TCP, dtype=float),
        r=SPHERE_R_MM,
        color=col
    )

    feasible_spheres.append(s)

o3d.visualization.draw_geometries(
    [pcd] + feasible_spheres,
    window_name='ALL Feasible TCPs'
)




vis_spheres = []
vis_points_for_center = []

for g, col in zip(subset_pair, colors_pair):
    # If this point is the target point, skip the regular ball (to avoid overlapping with the purple ball and causing Z-fighting).
    if target is not None and np.allclose(np.asarray(g.TCP, float), np.asarray(target.TCP, float)):
        continue
    s = sphere_at(g.TCP, SPHERE_R_MM, color=col)
    vis_spheres.append(s)
    vis_points_for_center.append(np.asarray(g.TCP, float))

# Set the viewpoint center (using all ordinary points; if empty, use the optimal point)
if vis_points_for_center:
    crop_center = np.mean(np.vstack(vis_points_for_center), axis=0)
elif target is not None:
    crop_center = np.asarray(target.TCP, float)
else:
    crop_center = np.asarray(pcd.get_center())

# ---- Global view + Gripper ----
vis1 = o3d.visualization.Visualizer()
vis1.create_window(window_name="Global view (single pair)["+TARGET+" case]: object pcd + gripper mesh + TCP", width=1200, height=800)
vis1.add_geometry(pcd)


for gobj in vis_spheres:
    vis1.add_geometry(gobj)


vis1.add_geometry(sphere_targetTCP)


for gobj in [pcd_tcp_alledge, frame_targetTCP] + gripper_mesh:
    if gobj is not None:
        vis1.add_geometry(gobj)

opt1 = vis1.get_render_option()
opt1.point_size = 1.0
opt1.mesh_show_back_face = True
ctr1 = vis1.get_view_control()
ctr1.set_lookat(crop_center.tolist())
ctr1.set_front([0.5, -0.5, -0.7])
ctr1.set_up([0, 0, 1])
ctr1.set_zoom(0.6)
vis1.run()
vis1.destroy_window()

# ---- Local view no Gripper ----
vis3 = o3d.visualization.Visualizer()
vis3.create_window(window_name="Local view (single pair)["+TARGET+" case]: object cloud + best TCP", width=1200, height=800)
vis3.add_geometry(pcd)

for gobj in vis_spheres:
    vis3.add_geometry(gobj)


vis3.add_geometry(sphere_targetTCP)
vis3.add_geometry(frame_targetTCP)

opt3 = vis3.get_render_option()
opt3.point_size = 1.0
opt3.mesh_show_back_face = True
ctr3 = vis3.get_view_control()
ctr3.set_lookat(crop_center.tolist())
ctr3.set_front([0.5, -0.5, -0.7])
ctr3.set_up([0, 0, 1])
ctr3.set_zoom(0.6)
vis3.run()
vis3.destroy_window()
# '''