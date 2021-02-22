import numpy as np
import torch
from pointnet.datasets import PoseTrainingDataset, PoseTestingDataset
from utils.file_utils import training_data_dir, testing_data_dir, load_pickle, \
    data_root_dir, testing_data_root, root_dir
from torch.utils.data import DataLoader, random_split
from pointnet.loss import PoseLoss
from pointnet.model import PointNetRot6d, FCNet, PointNetCls, PointNetRot9d, PointNetRot6d_Wide
from benchmark_utils.pose_evaluator import PoseEvaluator
import time
from tqdm import tqdm
from pointnet.prepare_data import unbatch_prediction
import json
import os
#'''
csv_path = testing_data_root + '/objects_v1.csv'
pose_evaluator = PoseEvaluator(csv_path)
object_names = load_pickle(data_root_dir + '/object_names.pkl')
def eval(pred, gt, object_id):
    R_pred = pred[:3,:3]
    t_pred = pred[:3, 3]
    R_gt = gt[:3,:3]
    t_gt = gt[:3,3]
    object_name = object_names[object_id]
    result = pose_evaluator.evaluate(object_name, R_pred, R_gt,t_pred,t_gt,np.ones(3))
    return result['rre_symmetry'], result['pts_err']

num_epochs = 10000
batch_size=256
save_freq=5
device = torch.device('cuda:1')

testing_dataset = PoseTestingDataset(data_root_dir+'/testing_object_pc')
testing_loader = DataLoader(testing_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

sym_Rs_np = load_pickle(data_root_dir + '/sym_Rs.pkl')
sym_Rs = [torch.from_numpy(sym_R_np).float().to(device) for sym_R_np in sym_Rs_np]
is_zinfs_np = load_pickle(data_root_dir + '/is_zinfs.pkl')
is_zinfs = torch.from_numpy(is_zinfs_np).to(device)
rot_axs_np = load_pickle(data_root_dir + '/rot_axs.pkl')
rot_axs = torch.from_numpy(rot_axs_np).float().to(device)
loss_func = PoseLoss(sym_Rs, is_zinfs, rot_axs)

#net=PointNetCls().to(device)
net = PointNetRot6d(channel=6).to(device)
#net=FCNet().to(device)
optimizer = torch.optim.Adam(net.parameters(), lr=0.001, betas=(0.9, 0.999))
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=40, gamma=0.5)

saved_model = torch.load(root_dir + '/saved_models/rgb_geodesic_v2_epoch135.pth')
net.load_state_dict(saved_model)
preds=[]
for iter_index, (pc, pc_mean) in enumerate(testing_loader):
    pc = pc.to(device)
    pred = net(pc).detach().cpu().numpy()
    pc_mean = pc_mean.numpy()
    #pred[:,:3,3]+=pc_mean
    preds.append(pred)
preds=np.concatenate(preds,axis=0)
print(preds.shape)
np.save(root_dir+'/nn_pred.npy', preds)
#'''
preds= np.load(root_dir+'/nn_pred.npy')
ans = unbatch_prediction(preds)

if os.path.exists("nn.json"):
    os.system('rm nn.json')
with open('nn.json', 'w') as f:
    json.dump(ans, f)