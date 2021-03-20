import numpy as np
import open3d as o3d
import copy
from scipy.spatial import KDTree

def naive_icp(source, target, threshold=0.1, initial_transformation=np.identity(4),
              max_iteration=100):
    reg_p2p = o3d.pipelines.registration.registration_icp(
        source, target, threshold, initial_transformation,
        o3d.pipelines.registration.TransformationEstimationPointToPlane(),
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=max_iteration))
    return reg_p2p.transformation

def naive_icp_colored(source, target, threshold=0.1, initial_transformation=np.identity(4),
              max_iteration=100):
    reg_p2p = o3d.pipelines.registration.registration_colored_icp(
        source, target, threshold, initial_transformation,
        o3d.pipelines.registration.TransformationEstimationForColoredICP(),
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=max_iteration))
    return reg_p2p.transformation

def colored_icp(source, target):
    voxel_radius = [0.04, 0.02, 0.01]
    max_iter = [50, 30, 14]
    current_transformation = np.identity(4)
    print("3. Colored point cloud registration")
    for scale in range(3):
        iter = max_iter[scale]
        radius = voxel_radius[scale]
        print([iter, radius, scale])

        print("3-1. Downsample with a voxel size %.2f" % radius)
        source_down = source.voxel_down_sample(radius)
        target_down = target.voxel_down_sample(radius)

        print("3-2. Estimate normal.")
        source_down.estimate_normals(
            o3d.geometry.KDTreeSearchParamHybrid(radius=radius * 2, max_nn=30))
        target_down.estimate_normals(
            o3d.geometry.KDTreeSearchParamHybrid(radius=radius * 2, max_nn=30))

        print("3-3. Applying colored point cloud registration")
        result_icp = o3d.pipelines.registration.registration_colored_icp(
            source_down, target_down, radius, current_transformation,
            o3d.pipelines.registration.TransformationEstimationForColoredICP(),
            o3d.pipelines.registration.ICPConvergenceCriteria(relative_fitness=1e-6,
                                                              relative_rmse=1e-6,
                                                              max_iteration=iter))
        current_transformation = result_icp.transformation
        print(result_icp)
    return current_transformation

def draw_registration_result_original_color(source, target, transformation):
    source_temp = copy.deepcopy(source)
    target = copy.deepcopy(target)
    source_temp.colors = o3d.utility.Vector3dVector(np.tile([255,0,0],
                                                       (np.asarray(source_temp.points).shape[0], 1)))
    source_temp.transform(transformation)
    o3d.visualization.draw_geometries([source_temp, target])
#                                      zoom=0.5,
#                                      front=[-0.2458, -0.8088, 0.5342],
#                                      lookat=[1.7745, 2.2305, 0.9787],
#                                      up=[0.3109, -0.5878, -0.7468])


def draw_registration_result_with_transformation(source, target, transformation):
    source = copy.deepcopy(source)
    target_temp = copy.deepcopy(target)
    translation = np.zeros_like(transformation)
    translation[:3,3]=np.array([0.2,0.2,0]).reshape(3,1)
    #target_temp.transform(transformation + translation)
    target_temp.transform(transformation)
    target_temp.transform(np.identity(4)+translation)
    o3d.visualization.draw_geometries([source, target_temp])

def draw_registration_result_red_inverse(source, target, transformation):
    source = copy.deepcopy(source)
    source.colors = o3d.utility.Vector3dVector(np.tile([255,0,0],
                                                       (np.asarray(source.points).shape[0], 1)))
    target_temp = copy.deepcopy(target)
    target_temp.transform(transformation)
    o3d.visualization.draw_geometries([source, target_temp])
#                                      zoom=0.5,
#                                      front=[-0.2458, -0.8088, 0.5342],
#                                      lookat=[1.7745, 2.2305, 0.9787],
#                                      up=[0.3109, -0.5878, -0.7468])

def draw_object_pc_with_filtered_points(source,mask):
    source = copy.deepcopy(source)
    pc_shape = np.asarray(source.points).shape[0]
    color_np = np.zeros([pc_shape, 3])
    color_np = np.broadcast_to(np.array([0,0,255]).reshape([1,-1]), color_np.shape).copy()
    color_np[mask] = np.broadcast_to(np.array([255,0,0]).reshape([1,-1]), color_np[mask].shape).copy()
    source.colors = o3d.utility.Vector3dVector(color_np)
    o3d.visualization.draw_geometries([source])

def get_distant_points_mask(point_np):
    center = np.mean(point_np, axis=0)
    dist = np.linalg.norm(point_np-center, axis=1)
    dist_mean = np.mean(dist)
    dist_std = np.std(dist)
    distant_points = (dist-dist_mean)>dist_std*2
    dist_mean_new = np.mean(dist[np.logical_not(distant_points)])
    dist_std_new = np.std(dist[np.logical_not(distant_points)])
    distant_points_new = (dist-dist_mean_new)>dist_std_new*1.5
    return np.bitwise_or(distant_points, distant_points_new)

def find_non_2d_boundary(label, object_id):
    object_idx = label == object_id
    h, w = label.shape
    object_idx_up = np.concatenate([label[1:,:], np.zeros([1,w])+80], axis=0) == object_id
    object_idx_up2 = np.concatenate([label[2:,:], np.zeros([2,w])+80], axis=0) == object_id
    object_idx_up3 = np.concatenate([label[3:,:], np.zeros([3,w])+80], axis=0) == object_id
    object_idx_down = np.concatenate([np.zeros([1,w])+80, label[:-1,:]], axis=0) == object_id
    object_idx_down2 = np.concatenate([np.zeros([2,w])+80, label[:-2,:]], axis=0) == object_id
    object_idx_left = np.concatenate([label[:,1:], np.zeros([h,1])+80], axis=1) == object_id
    object_idx_left2 = np.concatenate([label[:,2:], np.zeros([h,2])+80], axis=1) == object_id
    object_idx_right = np.concatenate([np.zeros([h,1])+80, label[:,:-1]], axis=1) == object_id
    object_idx_right2 = np.concatenate([np.zeros([h,2])+80, label[:,:-2]], axis=1) == object_id

    non_boundary = (object_idx * object_idx_up * object_idx_down * object_idx_left * object_idx_right
                    * object_idx_up2 * object_idx_down2 * object_idx_left2 * object_idx_right2)
    return non_boundary.flatten()

def remove_segmentation_outliers(source_points_np, non_boundary_mask):
    boundary_mask = np.bitwise_not(non_boundary_mask)
    if non_boundary_mask.sum()==0:
        object_geom_center = np.mean(source_points_np, axis=0)
    else:
        object_geom_center = np.mean(source_points_np[non_boundary_mask], axis=0)
    dist_to_center = np.linalg.norm(source_points_np - object_geom_center, axis=1)
    dist_mean = np.mean(dist_to_center)
    dist_std = np.std(dist_to_center)
    dist_mask = (dist_to_center-dist_mean) > 2*dist_std
    #boundary_dist_to_center = dist_to_center[boundary_mask]
    #ratio = 0.1
    #distance_threshold = np.sort(boundary_dist_to_center)[int(boundary_dist_to_center.shape[0] * ratio)]
    #mask = boundary_mask * (dist_to_center > distance_threshold)
    kdtree = KDTree(source_points_np)
    radius = 0.004
    nn_idx = kdtree.query_ball_point(source_points_np[boundary_mask], r=radius)
    for idx, nns in zip(np.where(boundary_mask)[0], nn_idx):
        if boundary_mask[nns].sum()<len(nns)*0.5:
            boundary_mask[idx]=False
    discard_mask = boundary_mask + dist_mask
    return np.bitwise_not(discard_mask)

def preprocess_point_cloud(pcd, voxel_size):
    #print(":: Downsample with a voxel size %.3f." % voxel_size)
    pcd_down = pcd.voxel_down_sample(voxel_size)

    radius_normal = voxel_size * 2
    #print(":: Estimate normal with search radius %.3f." % radius_normal)
    pcd_down.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))

    radius_feature = voxel_size * 5
    #print(":: Compute FPFH feature with search radius %.3f." % radius_feature)
    pcd_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        pcd_down,
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100))
    return pcd_down, pcd_fpfh

def execute_global_registration(source_down, target_down, source_fpfh,
                                target_fpfh, voxel_size):
    distance_threshold = voxel_size * 1.5
    #print(":: RANSAC registration on downsampled point clouds.")
    #print("   Since the downsampling voxel size is %.3f," % voxel_size)
    #print("   we use a liberal distance threshold %.3f." % distance_threshold)
    result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        source_down, target_down, source_fpfh, target_fpfh, True,
        distance_threshold,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
        3, [
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(
                0.9),
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(
                distance_threshold)
        ], o3d.pipelines.registration.RANSACConvergenceCriteria(100000, 0.999))
    return result

def apply_transformation(transformation, pts):
    R = transformation[0:3,0:3]
    t=transformation[0:3,3].squeeze()
    return (R@pts.transpose(1,0)).transpose(1,0) + t

def evaluate_registration(source, target, transformation):
    source_pts = np.asarray(source.points)
    source_colors = np.asarray(source.colors)
    target_pts = np.asarray(target.points)
    target_pts = apply_transformation(transformation, target_pts)
    target_colors = np.asarray(target.colors)
    target_kdtree = KDTree(target_pts)
    dist, idx = target_kdtree.query(source_pts)
    dist = dist.mean()
    color_dist = np.linalg.norm(source_colors - target_colors[idx], axis=1).mean()
    return dist, color_dist

def evaluate_registration_from_numpy(source, target, transformation):
    source_pts = np.asarray(source.points)
    source_colors = np.asarray(source.colors)
    target_pts = np.asarray(target.points)
    target_pts = apply_transformation(transformation, target_pts)
    target_colors = np.asarray(target.colors)
    target_kdtree = KDTree(target_pts)
    dist, idx = target_kdtree.query(source_pts)
    dist = dist.mean()
    color_dist = np.linalg.norm(source_colors - target_colors[idx], axis=1).mean()
    return dist, color_dist

