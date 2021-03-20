import numpy as np
import pandas as pd
import os
import open3d as o3d
from parse import parse
from PIL import Image
import copy
import json
import time
from pathos.multiprocessing import ProcessingPool as Pool
import sys

'''
from utils.file_utils import testing_data_root, load_pickle, data_root_dir, testing_data_final_dir
from benchmark_utils.pose_evaluator import PoseEvaluator

csv_path = testing_data_root + '/objects_v1.csv'
pose_evaluator = PoseEvaluator(csv_path)
object_names = load_pickle(data_root_dir + '/object_names.pkl')
def eval(pred, gt, object_id):
    pred=np.array(pred)
    gt=np.array(gt)
    R_pred = pred[:3,:3]
    t_pred = pred[:3, 3]
    R_gt = gt[:3,:3]
    t_gt = gt[:3, 3]

    object_name = object_names[object_id]
    result = pose_evaluator.evaluate(object_name, R_pred, R_gt,t_pred,t_gt,np.ones(3))
    return result['rre_symmetry'], result['pts_err']
'''
icp_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(icp_dir)
sys.path.append(root_dir)

from utils.file_utils import training_data_dir, testing_data_dir, split_dir,\
    get_split_files, get_data_files, load_pickle, testing_data_root, data_root_dir,\
    root_dir, utils_dir, testing_data_final_dir
from utils.preprocessing_utils import get_pc_from_image_files, sample_pc_from_mesh_file
from utils.visualize_utils import visualize_pc
from ICP import naive_icp, naive_icp_colored, draw_registration_result_original_color, \
    draw_registration_result_red_inverse, draw_registration_result_with_transformation, preprocess_point_cloud, \
    execute_global_registration, apply_transformation, evaluate_registration, \
    get_distant_points_mask, draw_object_pc_with_filtered_points, remove_segmentation_outliers, find_non_2d_boundary

def global_registration_proposal(source_down_points, source_down_colors,
                                 target_down_points, target_down_colors,
                                 source_fpfh, target_fpfh, voxel_size):
    source_down = o3d.geometry.PointCloud()
    source_down.points = o3d.utility.Vector3dVector(source_down_points)
    source_down.colors = o3d.utility.Vector3dVector(source_down_colors)
    target_down = o3d.geometry.PointCloud()
    target_down.points = o3d.utility.Vector3dVector(target_down_points)
    target_down.colors = o3d.utility.Vector3dVector(target_down_colors)
    source_feature = o3d.pipelines.registration.Feature()
    source_feature.data = source_fpfh
    target_feature = o3d.pipelines.registration.Feature()
    target_feature.data = target_fpfh
    initial_transformation = execute_global_registration(source_down, target_down,
                                                         source_feature, target_feature,
                                                         voxel_size)
    return initial_transformation.transformation

def build_source_target_pcds(source_points, source_colors,
                         target_points, target_colors, target_normals):
    source = o3d.geometry.PointCloud()
    source.points = o3d.utility.Vector3dVector(source_points)
    source.colors = o3d.utility.Vector3dVector(source_colors)
    target = o3d.geometry.PointCloud()
    target.points = o3d.utility.Vector3dVector(target_points)
    target.colors = o3d.utility.Vector3dVector(target_colors)
    target.normals = o3d.utility.Vector3dVector(target_normals)
    return source, target

def is_better_transformation(dist, color_dist, best_dist, best_color_dist):
    return dist<best_dist or dist<best_dist * 1.03 and color_dist<best_color_dist*0.9

def local_icp_refinement(source_points, source_colors,
                         target_points, target_colors, target_normals, initial_transformation):
    source, target = build_source_target_pcds(source_points, source_colors,
                         target_points, target_colors, target_normals)
    # object_geom_center_transformed = apply_transformation(initial_transformation, object_geom_center)
    # additional_translation = target_geom_center-object_geom_center_transformed
    # initial_transformation[0:3,3]+=additional_translation
    # print(initial_transformation)
    #draw_registration_result_original_color(source, target, initial_transformation)
    # print(initial_transformation)
    inverse_transformation = naive_icp(source, target, threshold=0.1,
                                       initial_transformation=initial_transformation,
                                       max_iteration=100)

    R = inverse_transformation[:3, :3]
    Rinv = np.linalg.inv(R)
    t = inverse_transformation[0:3, 3:]
    transformation = np.bmat([[Rinv, -Rinv @ t],
                              [np.zeros([1, 3]), np.eye(1)]])
    # print(transformation)
    # print('------')
    # gt_transformation = meta['poses_world'][object_id]
    # print(gt_transformation)
    #draw_registration_result_original_color_inverse(object_pcd, target_pcd, transformation)
    # draw_registration_result_original_color_inverse(object_pcd, target_pcd, gt_transformation)

    # exit()
    dist, color_dist = evaluate_registration(source, target, transformation)
    return transformation, dist, color_dist

def parallel_run_icp_refinement(pool,
                                source_points_np, source_colors_np,
                                target_points_np, target_colors_np,
                                target_normals_np, initial_poses):
    source_point_data = [source_points_np for i in range(len(initial_poses))]
    source_color_data = [source_colors_np for i in range(len(initial_poses))]
    target_point_data = [target_points_np for i in range(len(initial_poses))]
    target_color_data = [target_colors_np for i in range(len(initial_poses))]
    target_normal_data = [target_normals_np for i in range(len(initial_poses))]
    initial_transformation_data = initial_poses
    registration_outputs = pool.map(local_icp_refinement,
                                    source_point_data, source_color_data,
                                    target_point_data, target_color_data,
                                    target_normal_data,
                                    initial_transformation_data)

    best_transformation, _, _ = registration_outputs[0]
    best_dist = 10000
    best_color_dist = 10000
    for i in range(len(initial_poses)):
        # print('iter %d'%i)
        transformation, dist, color_dist = registration_outputs[i]
        if (is_better_transformation(dist, color_dist, best_dist, best_color_dist)):
            best_dist = dist
            best_color_dist = color_dist
            best_transformation = transformation
    return best_dist, best_color_dist, best_transformation

np.random.seed(int(time.time()))
csv_data = pd.read_csv(testing_data_root+'/objects_v1.csv')
gt_pcs = []
gt_pc_root = data_root_dir + '/model_pc'
classname2idx = {}
npoints = 2048
for i in range(len(csv_data)):
    print('processing model #%d'%i)
    object_entry = csv_data.loc[i]
    object_location = object_entry['location']
    object_name = parse("{}/{}", object_location)[1]
    classname2idx[object_entry['class']] = i
    gt_pc_file = gt_pc_root+'/%s.ply'%object_name
    if not os.path.exists(gt_pc_file):
        mesh_file = os.path.join(data_root_dir, object_entry['location']) + '/visual_meshes/visual.dae'
        texture_file = os.path.join(data_root_dir, object_entry['location']) + '/visual_meshes/texture_map.png'
        pcd = sample_pc_from_mesh_file(mesh_file, texture_file, npoints)
        o3d.io.write_point_cloud(gt_pc_file, pcd)
        gt_pcs.append(pcd)
    else:
        pcd = o3d.io.read_point_cloud(gt_pc_file)
        gt_pcs.append(pcd)
testing_pc_root = data_root_dir + '/testing_data_final_object_pc'
rgb_files, depth_files, label_files, meta_files = get_data_files(testing_data_final_dir,
                                                                 target_levels=(1,2,3))
#testing_pc_root = data_root_dir + '/training_pc'
#rgb_files, depth_files, label_files, meta_files = get_data_files(training_data_dir,
#                                                                 target_levels=(1,2,3))
#testing_pc_root = data_root_dir + '/testing_pc'
#rgb_files, depth_files, label_files, meta_files = get_data_files(testing_data_dir,
#                                                                 target_levels=(1,2))
testing_pcs = []

ans = {}
counter=0
object_counter = 0
correct_object_counter=0

num_processes = 16
num_initial_poses = 256
pool = Pool(num_processes)
target_instance_name = '2-224-11'
target_object_id=76
Rs = load_pickle(data_root_dir + '/sym_Rs.pkl')[-1]
#rgb_files = rgb_files[457:]
#depth_files = depth_files[457:]
#label_files = label_files[457:]
#meta_files = meta_files[457:]
for rgb_file, depth_file, label_file, meta_file in zip(rgb_files, depth_files,
                                                       label_files, meta_files):
    start_time=time.time()
    instance_name = parse("{}/v2.2/{}_meta.pkl", meta_file)[1]
    #if instance_name!=target_instance_name:
    #    continue
    #print('processing #%d %s'%(counter, instance_name))
    counter+=1
    #if(counter>1):
    #    break
    testing_pc_file = testing_pc_root + '/%s.ply'%instance_name
    testing_pc = get_pc_from_image_files(rgb_file, depth_file, meta_file)
    #if not os.path.exists(testing_pc_file):
    #    print(np.asarray(testing_pc.points).shape)
    #    o3d.io.write_point_cloud(testing_pc_file, testing_pc)
    #else:
    #    testing_pc = o3d.io.read_point_cloud(testing_pc_file)

    label = np.array(Image.open(label_file))
    meta = load_pickle(meta_file)

    scales = meta['scales']
    object_ids = meta['object_ids']
    intrinsic = meta['intrinsic']
    extrinsic = meta['extrinsic']
    ans_scene = {}
    poses = []
    #if counter==1:
    #    continue
    for object_id, scale in enumerate(scales):
        if scale is None:
            poses.append(None)
            continue
        object_counter+=1
        #if object_id!=target_object_id:
        #    continue

        object_begin_time = time.time()
        print('scene: %s, object id: '%instance_name, object_id)

        scale = scales[object_id]
        point_idx = label.flatten()==object_id
        testing_pts = np.asarray(testing_pc.points)
        testing_colors = np.asarray(testing_pc.colors)

        object_pcd = o3d.geometry.PointCloud()
        source_points_np = testing_pts[point_idx]
        source_colors_np = testing_colors[point_idx]
        if(source_points_np.shape[0]==0):
            poses.append(np.identity(4).tolist())
            continue
        #mask = np.logical_not(get_distant_points_mask(source_points_np))
        non_boundary_mask = find_non_2d_boundary(label, object_id)[point_idx]
        mask = remove_segmentation_outliers(source_points_np, non_boundary_mask)
        if (mask.sum()==0):
            mask = np.ones_like(mask, dtype=np.bool)
        source_points_np = source_points_np[mask]
        source_colors_np = source_colors_np[mask]
        if source_points_np.shape[0]>2048:
            random_idx = np.random.choice(source_points_np.shape[0], 2048, replace=False)
            source_points_np = source_points_np[random_idx]
            source_colors_np = source_colors_np[random_idx]
        source_points = o3d.utility.Vector3dVector(source_points_np)
        source_colors = o3d.utility.Vector3dVector(source_colors_np)
        #print('object pcd points: ', testing_pts[point_idx].shape[0])
        object_pcd.points = source_points
        object_pcd.colors = source_colors
        object_geom_center = np.asarray(source_points).mean(axis=0)
        #draw_object_pc_with_filtered_points(object_pcd, mask)

        target_pcd = copy.deepcopy(gt_pcs[object_id])
        #print('target pcd points: ', np.asarray(target_pcd.points).shape[0])
        target_points_np = np.asarray(target_pcd.points)*scale
        target_colors_np = np.asarray(target_pcd.colors)
        target_normals_np = np.asarray(target_pcd.normals)
        target_points = o3d.utility.Vector3dVector(np.asarray(target_pcd.points)*scale)
        target_colors = target_pcd.colors
        target_pcd.points=target_points
        target_geom_center = np.asarray(target_points).mean(axis=0)

        voxel_size = (target_points_np.max(axis=0) - target_points_np.min(axis=0)).min() / 5
        source_down, source_fpfh = preprocess_point_cloud(object_pcd, voxel_size/2)
        target_down, target_fpfh = preprocess_point_cloud(target_pcd, voxel_size/2)
        source_down_points_np = np.asarray(source_down.points)
        source_down_colors_np = np.asarray(source_down.colors)
        target_down_points_np = np.asarray(target_down.points)
        target_down_colors_np = np.asarray(target_down.colors)
        #print(type(source_fpfh))
        #exit()
        source_fpfh = source_fpfh.data
        target_fpfh = target_fpfh.data

        source_down_point_data = [source_down_points_np for i in range(num_initial_poses)]
        source_down_color_data = [source_down_colors_np for i in range(num_initial_poses)]
        target_down_point_data = [target_down_points_np for i in range(num_initial_poses)]
        target_down_color_data = [target_down_colors_np for i in range(num_initial_poses)]
        source_fpfh_data = [source_fpfh for i in range(num_initial_poses)]
        target_fpfh_data = [target_fpfh for i in range(num_initial_poses)]
        voxel_size_data = [voxel_size for i in range(num_initial_poses)]
        initial_poses = pool.map(global_registration_proposal,
                                 source_down_point_data, source_down_color_data,
                                 target_down_point_data, target_down_color_data,
                                 source_fpfh_data, target_fpfh_data, voxel_size_data)
        pruned_initial_poses = []
        for pose in initial_poses:
            is_unique=True
            for unique_pose in pruned_initial_poses:
                if np.linalg.norm(pose-unique_pose, ord='fro')<0.5:
                    is_unique=False
                    break
            if is_unique:
                pruned_initial_poses.append(pose)
        if len(pruned_initial_poses)>16:
            #print('len pruned pose > 16', len(pruned_initial_poses))
            pruned_initial_poses = pruned_initial_poses[:16]
        #print('initial pose time: ', time.time()- object_begin_time)
        icp_begin_time = time.time()
        #print('unique poses: ', len(pruned_initial_poses))
        #exit()
        best_dist, best_color_dist, best_transformation = \
            parallel_run_icp_refinement(pool,
                                        source_points_np, source_colors_np,
                                        target_points_np, target_colors_np,
                                        target_normals_np, pruned_initial_poses)

        #draw_registration_result_red_inverse(object_pcd, target_pcd, best_transformation)
        sym_poses = []
        #print('best_transformation: ', best_transformation)
        for R in Rs:
            sym_transformation = best_transformation
            sym_transformation[:3,:3] = sym_transformation[:3,:3] @ R
            #draw_registration_result_red_inverse(object_pcd, target_pcd, sym_transformation)
            sym_poses.append(np.linalg.inv(sym_transformation))
        #print(best_transformation)
        best_dist, best_color_dist, best_transformation = \
            parallel_run_icp_refinement(pool,
                                        source_points_np, source_colors_np,
                                        target_points_np, target_colors_np,
                                        target_normals_np, sym_poses)
        #print('icp time: ', time.time()-icp_begin_time)
        #print('single object time: ', time.time()-object_begin_time)
        #print('extent: ', target_points_np.max(axis=0)-target_points_np.min(axis=0))
        #print('voxel_size: ',(target_points_np.max(axis=0)-target_points_np.min(axis=0)).min()/5)

        #draw_registration_result_red_inverse(object_pcd, target_pcd, best_transformation)
        best_transformation = np.linalg.inv(extrinsic) @ best_transformation
        #degerr, pterr = eval(best_transformation, meta['poses_world'][object_id], object_id)
        #if(degerr<5 and pterr<0.01):
        #    correct_object_counter+=1
        #print('degerr=%f, pterr=%f'%(degerr, pterr))
        #draw_registration_result_with_transformation(object_pcd, target_pcd, best_transformation)
        #exit()
        poses.append(best_transformation.tolist())
    ans_scene['poses_world'] = poses
    ans[instance_name] = ans_scene
    #visualize_pc(testing_pc)
    print('instance #%d: %f s used'%(counter, time.time()-start_time))
    with open(data_root_dir+'/icp_global_final.json', 'w') as f:
        json.dump(ans, f)
#print(object_counter)
#print('acc: ', correct_object_counter * 1.0 / object_counter)
pool.close()
pool.join()

