import numpy as np
import pandas as pd
import os
from parse import parse
from PIL import Image
import sys

nn_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(nn_dir)
sys.path.append(root_dir)

from utils.file_utils import training_data_dir, testing_data_dir, split_dir,\
    get_split_files, get_data_files, load_pickle, testing_data_root, data_root_dir,\
    root_dir, utils_dir, save_pickle
from utils.preprocessing_utils import get_pc_from_image_files, sample_pc_from_mesh_file
from utils.visualize_utils import visualize_pc
from benchmark_utils.pose_evaluator import PoseEvaluator

def process_symmetries():
    csv_path = testing_data_root+'/objects_v1.csv'
    pose_evaluator = PoseEvaluator(csv_path)
    object_db = pose_evaluator.objects_db
    csv_data=pd.read_csv(csv_path)
    rs = []
    is_zinfs = []
    object_names=[]
    rot_axs = []
    for i in range(len(csv_data)):
        object_name = csv_data.loc[i]['object']
        object_names.append(object_name)
        object_entry = object_db[object_name]
        sym_rots = object_entry['sym_rots']
        rot_axis = object_entry['rot_axis']
        # sym_rots: nsym x 3 x 3
        rs.append(sym_rots)
        is_zinfs.append(rot_axis is not None)
        rot_axs.append(rot_axis if rot_axis is not None else np.array([0,0,0]))
    is_zinfs = np.array(is_zinfs)
    rot_axs = np.stack(rot_axs)
    save_pickle(data_root_dir + '/sym_Rs.pkl', rs)
    save_pickle(data_root_dir + '/is_zinfs.pkl', is_zinfs)
    save_pickle(data_root_dir + '/object_names.pkl', object_names)
    save_pickle(data_root_dir + '/rot_axs.pkl', rot_axs)

def process_scene_images(image_data_dir, is_training=True, subset=False, subset_scenes=100):
    mode = "training" if is_training else 'testing'
    if subset:
        mode = mode+'_subset'
    rgb_files, depth_files, label_files, meta_files = get_data_files(image_data_dir,
                                                                     target_levels=(1,2))
    object_pc_dir = data_root_dir + '/%s_object_pc'%mode
    pc_files = []
    Rs = []
    is_bads = []
    model_ids=[]
    pc_means=[]
    npoints=1024
    counter=0
    for rgb_file, depth_file, label_file, meta_file in zip(rgb_files, depth_files,
                                                           label_files, meta_files):
        counter+=1
        if subset and counter>subset_scenes:
            break
        print('processing scene #%d'%counter)
        instance_name = parse("{}/v2.2/{}_meta.pkl", meta_file)[1]

        current_pc = get_pc_from_image_files(rgb_file, depth_file, meta_file)
        current_pts = np.asarray(current_pc.points)
        current_colors= np.asarray(current_pc.colors)
        label = np.array(Image.open(label_file))
        meta = load_pickle(meta_file)

        scales = meta['scales']
        is_bad=[]
        if is_training:
            gt_poses = meta['poses_world']
        for object_id, scale in enumerate(scales):
            if scale is None:
                is_bad.append(None)
                continue
            model_ids.append(object_id)

            scale = scales[object_id]
            point_idx = label.flatten()==object_id

            object_pts = np.asarray(current_pts[point_idx])
            object_colors = np.asarray(current_colors[point_idx])
            if(object_pts.shape[0]==0):
                is_bad.append(True)
                continue
            else:
                is_bad.append(False)

            object_pts /= scale
            pts_mean = object_pts.mean(axis=0)
            object_pts = (object_pts-pts_mean)
            object_pts = np.concatenate([object_pts, object_colors], axis=1)

            current_npoints = object_pts.shape[0]
            if current_npoints>npoints:
                rand_idx = np.random.choice(current_npoints, npoints)
                object_pts = object_pts[rand_idx]
            else:
                pad_num = npoints-current_npoints
                object_pts = np.concatenate([object_pts, np.tile(object_pts[-1], (pad_num,1))])

            pc_file = object_pc_dir + '/%s_no%d.npy'%(instance_name, object_id)
            pc_files.append(pc_file)
            np.save(pc_file, object_pts)
            pc_means.append(pts_mean)
            if is_training:
                pose = gt_poses[object_id]
                pose[:3,3] = pose[:3,3]/scale - pts_mean
                Rs.append(pose)
        is_bads.append(is_bad)
    save_pickle(object_pc_dir + '/pc_filenames.pkl', pc_files)
    save_pickle(object_pc_dir + '/is_bads.pkl', is_bads)
    save_pickle(object_pc_dir + '/model_ids.pkl', model_ids)
    save_pickle(object_pc_dir + '/pc_means.pkl', pc_means)
    if is_training:
        save_pickle(object_pc_dir + '/gt_poses.pkl', Rs)

def unbatch_prediction(pred):
    is_bads = load_pickle(data_root_dir + '/testing_object_pc/is_bads.pkl')
    pc_means = load_pickle(data_root_dir + '/testing_object_pc/pc_means.pkl')
    rgb_files, depth_files, label_files, meta_files = get_data_files(testing_data_dir,
                                                                     target_levels=(1,2))
    current_pred_index=0
    ans = {}

    for is_bad, meta_file in zip(is_bads,meta_files):
        instance_name = parse("{}/v2.2/{}_meta.pkl", meta_file)[1]
        meta = load_pickle(meta_file)

        scales = meta['scales']
        ans_scene = {}
        poses=[]
        for object_id, scale in enumerate(scales):
            if scale is None:
                poses.append(None)
                continue
            if is_bad[object_id]:
                poses.append(np.identity(4).tolist())
            else:
                current_pred = pred[current_pred_index]
                #print('scale=', scale)
                #print('pred t=', current_pred[:3,3])
                #print('mean=',pc_means[current_pred_index])
                current_pred[:3,3]=(current_pred[:3,3]+pc_means[current_pred_index])*scale
                poses.append(current_pred.tolist())
                current_pred_index+=1
        ans_scene['poses_world'] = poses
        ans[instance_name] = ans_scene
    return ans

if __name__=='__main__':
    process_symmetries()
    #process_scene_images(training_data_dir, is_training=True, subset=False)
    #process_scene_images(training_data_dir, is_training=True, subset=True, subset_scenes=1)
    #process_scene_images(testing_data_dir, False)