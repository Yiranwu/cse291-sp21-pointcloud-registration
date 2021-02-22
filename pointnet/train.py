import numpy as np
import torch
import time
from tqdm import tqdm
import os
import sys

nn_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(nn_dir)
sys.path.append(root_dir)

from pointnet.datasets import PoseTrainingDataset, PoseTestingDataset
from utils.file_utils import training_data_dir, testing_data_dir, load_pickle, \
    data_root_dir, testing_data_root, root_dir
from torch.utils.data import DataLoader, random_split
from pointnet.loss import PoseLoss
from pointnet.model import PointNetRot6d, FCNet, PointNetCls, PointNetRot9d, PointNetRot6d_Wide
from benchmark_utils.pose_evaluator import PoseEvaluator
from benchmark_utils.pose_utils import compute_rre_symmetry

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
device = torch.device('cuda:0')

training_dataset = PoseTrainingDataset(data_root_dir+'/training_object_pc')
training_instances = len(training_dataset)
training_size = int(training_instances*0.8)
validation_size=training_instances-training_size
training_dataset, validation_dataset = random_split(training_dataset, [training_size, validation_size])
training_loader = DataLoader(training_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
validation_loader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
print('training dataset size: ',len(training_dataset))
#exit()
testing_dataset = PoseTestingDataset(data_root_dir+'/testing_object_pc')
testing_loader = DataLoader(testing_dataset, batch_size=batch_size, shuffle=True, num_workers=4)

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

def eval_model():
    saved_model = torch.load(root_dir + '/saved_models/rgb_geodesic_v2_epoch135.pth')
    net.load_state_dict(saved_model)
    rres = []
    pres = []
    fros = []
    for iter_index, (pc, gt_pose, model_id) in enumerate(validation_loader):
        pc = pc.to(device)
        gt_pose = gt_pose.to(device)
        pred = net(pc).detach().cpu().numpy()
        gt_pose = gt_pose.cpu().numpy()
        model_id = model_id.cpu().numpy()
        for pred_ele, gt_pose_ele, model_id_ele in zip(pred, gt_pose, model_id):
            rre, pre = eval(pred_ele, gt_pose_ele, model_id_ele)
            rres.append(rre)
            pres.append(pre)
    rres = np.array(rres)
    pres = np.array(pres)
    print(((rres<5) & (pres<0.01)).sum()/rres.shape[0])
    print(((rres<7) & (pres<0.01)).sum()/rres.shape[0])
    print((rres<10).sum()/rres.shape[0])
    print((rres<20).sum()/rres.shape[0])
    print((rres<40).sum()/rres.shape[0])
    print((rres<90).sum()/rres.shape[0])
    exit()
#eval_model()

for epoch in range(num_epochs):
    losses=[]
    epoch_start_time = time.time()
    for iter_index, (pc, gt_pose, model_id) in tqdm(enumerate(training_loader)):
    #for iter_index, (pc, gt_pose, model_id) in enumerate(training_loader):
        optimizer.zero_grad()
        # check type of is_zinf and model_id: is them correcly batched into torch tensors?
        pc=pc.to(device)
        gt_pose=gt_pose.to(device)

        pred = net(pc)
        #print('pred:',pred[:2])
        #print('gt:', gt_pose[:2])
        loss = loss_func(pred, gt_pose, model_id,with_symmetry=True)
        losses.append(loss.item())
        loss.backward()
        optimizer.step()
    scheduler.step()
    print('epoch %d, time %f, loss=%f'%(epoch,time.time()-epoch_start_time, sum(losses)/len(losses)))
    if epoch%save_freq==0:
        rres = []
        pres = []
        preds=[]
        gts=[]
        model_ids=[]
        training_rre_time = time.time()
        for iter_index, (pc, gt_pose, model_id) in enumerate(training_loader):
            pc=pc.to(device)
            gt_pose=gt_pose.to(device)
            pred = net(pc).detach().cpu().numpy()
            gt_pose = gt_pose.cpu().numpy()
            model_id = model_id.cpu().numpy()
            for pred_ele, gt_pose_ele, model_id_ele in zip(pred, gt_pose, model_id):
                rre,pre = eval(pred_ele, gt_pose_ele, model_id_ele)
                rres.append(rre)
                pres.append(pre)
                preds.append(pred_ele)
                gts.append(gt_pose_ele)
                model_ids.append(model_id_ele)
        #preds=np.stack(preds)
        #gts=np.stack(gts)
        #model_ids = np.stack(model_ids)
        #deg_bad_pos=np.where(np.array(rres)>10)[0]
        #print('@train: pos that deg err>10: %s, value:%s'%(deg_bad_pos, np.array(rres)[deg_bad_pos]))
        #print('bad pred:', preds[deg_bad_pos])
        #print('bad gt:', gts[deg_bad_pos])
        #print('bad model id:', model_ids[deg_bad_pos])
        #bad_preds_torch = torch.from_numpy(preds[deg_bad_pos])[:,:3,:3].to(device)
        #bad_gts_torch = torch.from_numpy(gts[deg_bad_pos])[:,:3,:3].to(device)
        #bad_ids_torch = torch.from_numpy(model_ids[deg_bad_pos]).to(device)
        #print('zloss for bad case: ',loss_func.R_zinf_loss(bad_preds_torch,bad_gts_torch, bad_ids_torch))

        print('training avg rre: %f, avg pre: %f, time:%f'%(sum(rres)/len(rres), sum(pres)/len(pres), training_rre_time-time.time()))
        rres = []
        pres = []
        fros = []
        for iter_index, (pc, gt_pose, model_id) in enumerate(validation_loader):
            pc=pc.to(device)
            gt_pose=gt_pose.to(device)
            pred = net(pc).detach().cpu().numpy()
            gt_pose = gt_pose.cpu().numpy()
            model_id = model_id.cpu().numpy()
            for pred_ele, gt_pose_ele, model_id_ele in zip(pred, gt_pose, model_id):
                rre,pre = eval(pred_ele, gt_pose_ele, model_id_ele)
                rres.append(rre)
                pres.append(pre)
        print('validation avg rre: %f, avg pre: %f'%(sum(rres)/len(rres), sum(pres)/len(pres)))
    if epoch%save_freq==0:
        torch.save(net.state_dict(), root_dir + '/saved_models/rgb_geodesic_sym_epoch%d.pth'%epoch)




