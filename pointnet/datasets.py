from torch.utils.data import Dataset, DataLoader
from utils.file_utils import load_pickle, data_root_dir
import numpy as np
import torch
from pointnet.loss import PoseLoss

class PoseTrainingDataset(Dataset):
    def __init__(self, data_path):
        super(PoseTrainingDataset, self).__init__()
        self.pc_files = load_pickle(data_path + '/pc_filenames.pkl')
        self.gt_poses = load_pickle(data_path + '/gt_poses.pkl')
        self.model_ids = load_pickle(data_path + '/model_ids.pkl')
        self.pc_means = load_pickle(data_path + '/pc_means.pkl')

    def __len__(self):
        return len(self.pc_files)

    def __getitem__(self, idx):
        pc = torch.from_numpy(np.load(self.pc_files[idx])).transpose(0,1).float()
        gt_pose = self.gt_poses[idx]
        #gt_pose[0:3,3]-=self.pc_means[idx]
        gt_pose = torch.from_numpy(gt_pose).float()
        #gt_pose = torch.from_numpy(np.identity(4)).float()
        #gt_pose = torch.rand([4,4]).float()
        model_id = self.model_ids[idx]
        return pc, gt_pose, model_id

class PoseTestingDataset(Dataset):
    def __init__(self, data_path):
        super(PoseTestingDataset, self).__init__()
        self.pc_files = load_pickle(data_path + '/pc_filenames.pkl')
        self.model_ids = load_pickle(data_path + '/model_ids.pkl')
        self.pc_means = load_pickle(data_path + '/pc_means.pkl')

    def __len__(self):
        return len(self.pc_files)

    def __getitem__(self, idx):
        pc = torch.from_numpy(np.load(self.pc_files[idx])).transpose(0,1).float()
        pc_mean = self.pc_means[idx]
        return pc, pc_mean

if __name__=='__main__':
    training_dataset = PoseTrainingDataset(data_root_dir+'/training_object_pc')