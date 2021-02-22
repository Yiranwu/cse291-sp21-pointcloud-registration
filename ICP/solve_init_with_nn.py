import numpy as np
from utils.file_utils import training_data_dir, testing_data_dir, split_dir,\
    get_split_files, get_data_files, load_pickle, testing_data_root, data_root_dir
from utils.preprocessing_utils import get_pc_from_image_files, sample_pc_from_mesh_file
from utils.visualize_utils import visualize_pc
import pandas as pd
import os
import open3d as o3d
from parse import parse
from PIL import Image
import copy
import json
from ICP import naive_icp, colored_icp, draw_registration_result_original_color, draw_registration_result_original_color_inverse

csv_data = pd.read_csv(testing_data_root+'/objects_v1.csv')
gt_pcs = []
gt_pc_root = data_root_dir + '/model_pc'
classname2idx = {}
npoints = 2048
for i in range(len(csv_data)):
    #print('processing model #%d'%i)
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


with open('../nn.json', 'r') as f:
    nn_preds = json.load(f)
#testing_pc_root = data_root_dir + '/training_pc'
#rgb_files, depth_files, label_files, meta_files = get_data_files(training_data_dir,
#                                                                 target_levels=(1,2))
testing_pc_root = data_root_dir + '/testing_pc'
rgb_files, depth_files, label_files, meta_files = get_data_files(testing_data_dir,
                                                                 target_levels=(1,2))
testing_pcs = []

ans = {}
counter=0
for rgb_file, depth_file, label_file, meta_file in zip(rgb_files, depth_files,
                                                       label_files, meta_files):

    instance_name = parse("{}/v2.2/{}_meta.pkl", meta_file)[1]
    nn_pred = nn_preds[instance_name]['poses_world']
    print('processing #%d %s'%(counter, instance_name))
    counter+=1
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
    ans_scene = {}
    poses = []
    for object_id, scale in enumerate(scales):
        if scale is None:
            poses.append(None)
            continue

        scale = scales[object_id]
        point_idx = label.flatten()==object_id
        testing_pts = np.asarray(testing_pc.points)
        testing_colors = np.asarray(testing_pc.colors)

        object_pcd = o3d.geometry.PointCloud()
        points = o3d.utility.Vector3dVector(testing_pts[point_idx])
        colors = o3d.utility.Vector3dVector(testing_colors[point_idx])
        #print('object pcd points: ', testing_pts[point_idx].shape[0])
        object_pcd.points = points
        object_pcd.colors = colors
        '''
        if(np.asarray(points).shape[0]==0):
            print(np.asarray(points).shape)
            print(object_id)
            print(object_ids)
            print(label.dtype)
            print(label)
            print(meta['poses_world'][object_id])
            print(np.where(point_idx))
            exit()
        '''
        object_geom_center = np.asarray(points).mean(axis=0)

        target_pcd = copy.deepcopy(gt_pcs[object_id])
        #print('target pcd points: ', np.asarray(target_pcd.points).shape[0])
        target_pcd.points = o3d.utility.Vector3dVector(np.asarray(target_pcd.points)*scale)
        target_geom_center = np.asarray(target_pcd.points).mean(axis=0)

        initial_transformation = np.array(nn_pred[object_id])
        #visualize_pc(object_pcd)
        #visualize_pc(target_pcd)
        #draw_registration_result_original_color(object_pcd, target_pcd, np.identity(4))
        #transformation = colored_registration(object_pcd, target_pcd)
        inverse_transformation = naive_icp(object_pcd, target_pcd, 0.05, initial_transformation)
        R = inverse_transformation[:3,:3]
        Rinv = np.linalg.inv(R)
        t = inverse_transformation[0:3, 3:]
        transformation = np.bmat([[Rinv, -Rinv@t],
                                  [np.zeros([1,3]),np.eye(1)]])
        #print(transformation)
        #print('------')
        #gt_transformation = meta['poses_world'][object_id]
        #print(gt_transformation)
        #draw_registration_result_original_color_inverse(object_pcd, target_pcd, transformation)
        #draw_registration_result_original_color_inverse(object_pcd, target_pcd, gt_transformation)

        #exit()

        poses.append(transformation.tolist())
    ans_scene['poses_world'] = poses
    ans[instance_name] = ans_scene
    #visualize_pc(testing_pc)

with open('icp_nn.json', 'w') as f:
    json.dump(ans, f)


