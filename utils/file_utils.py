#%%
import numpy as np
import matplotlib.pyplot as plt
import os
import pickle
from parse import parse

def load_pickle(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)

#data_root_dir = "/home/yiran/pc_mapping/HW2/data"
utils_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(utils_dir)
data_root_dir = root_dir +'/data'
training_data_root = data_root_dir + "/training_data"
training_data_dir = training_data_root + "/v2.2"
testing_data_root = data_root_dir + "/testing_data"
testing_data_dir = testing_data_root + "/v2.2"
split_dir = data_root_dir + "/training_data/splits/v2"

def get_split_files(split_name):
    return append_prefix_to_data_files(dir_name=split_dir, lookup_table_filename=split_dir + '/%s.txt' % split_name)

def append_prefix_to_data_files(dir_name, lookup_table_filename = None):
    if lookup_table_filename is None:
        lookup_table_filename = dir_name + "/lookup_table.txt"
    with open(lookup_table_filename, 'r') as f:
        prefix = [os.path.join(dir_name, line.strip()) for line in f if line.strip()]
        rgb = [p + "_color_kinect.png" for p in prefix]
        depth = [p + "_depth_kinect.png" for p in prefix]
        label = [p + "_label_kinect.png" for p in prefix]
        meta = [p + "_meta.pkl" for p in prefix]
    return rgb, depth, label, meta

def generate_file_lookup_table(dir_name, target_levels = (1,2)):
    file_names = []
    for fname in os.listdir(dir_name):
        if not fname[0].isdigit():
            continue
        result = parse("{}-{}-{}_{}.{}", fname)
        level_id, scene_id, variant_id, suffix, ext_name = result
        if suffix=='meta' and int(level_id) in target_levels:
            file_names.append("%s-%s-%s"%(level_id, scene_id, variant_id))
    with open(dir_name + "/lookup_table.txt", "w") as f:
        for file_name in file_names:
            f.write(file_name+'\n')

def get_data_files(dir_name, target_levels = (1,2)):
    lookup_table_filename = dir_name + "/lookup_table.txt"
    if not os.path.exists(lookup_table_filename):
        generate_file_lookup_table(dir_name, target_levels)
    return append_prefix_to_data_files(dir_name, lookup_table_filename=None)

