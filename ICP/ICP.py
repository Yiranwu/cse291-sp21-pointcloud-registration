import numpy as np
import open3d as o3d
import copy
from scipy.spatial import KDTree

def naive_icp(source, target, threshold=0.1, initial_transformation=np.identity(4),
              max_iteration=100):
    reg_p2p = o3d.pipelines.registration.registration_icp(
        source, target, threshold, initial_transformation,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(),
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


def draw_registration_result_original_color_inverse(source, target, transformation):
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

