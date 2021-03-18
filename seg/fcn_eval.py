import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch.nn.functional as F
import numpy as np
from seg.fcn_model import *
import os
import sys
from segmentation.models import all_models

seg_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(seg_dir)
sys.path.append(root_dir)

from utils.file_utils import training_data_dir, training_image_feature_dir
from seg.datasets import RGBTrainingDataset
from seg.seg_utils import class_weights
from benchmark_pose_and_detection.sem_seg_evaluator import Evaluator

model_name = "fcn8_resnet101"
device = torch.device('cuda:0')
batch_size = 4
n_classes = 82
num_epochs = 100
image_axis_minimum_size = 200
pretrained = True
fixed_feature = False

training_dataset = RGBTrainingDataset(training_data_dir, image_axis_minimum_size, subset=True)
training_loader = DataLoader(training_dataset, batch_size=batch_size, shuffle=True, num_workers=8)
training_dataset_origin_label = RGBTrainingDataset(training_data_dir, image_axis_minimum_size, subset=True, label_downsample=True)
training_loader_origin_label = DataLoader(training_dataset_origin_label, batch_size=batch_size, shuffle=True, num_workers=8)
### Model
net = fcn8_resnet101(n_classes, batch_size,
                     pretrained=pretrained,
                     fixed_feature=fixed_feature)
state_dict = torch.load(root_dir + '/saved_models/seg/fcn8_resnet101_epoch40_step1000.pth')
#print(state_dict.keys())
for key in list(state_dict.keys()):
    state_dict[key.replace('module.', '')] = state_dict.pop(key)
#print(state_dict.keys())
net.load_state_dict(state_dict)
net.to(device)

### Optimizers
if pretrained and fixed_feature:  # fine tunning
    params_to_update = net.parameters()
    print("Params to learn:")
    params_to_update = []
    for name, param in net.named_parameters():
        if param.requires_grad == True:
            params_to_update.append(param)
            print("\t", name)
    optimizer = torch.optim.Adadelta(params_to_update)
else:
    optimizer = torch.optim.Adam(net.parameters(), lr=0.001)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3000, gamma=0.4)


evaluator = Evaluator()
for step, (datas, labels, _) in tqdm(enumerate(training_loader_origin_label)):
    datas = datas.to(device)
    labels = labels.to(device)
    preds = net(datas)
    #preds = F.interpolate(preds, size=[720,1280], mode='bilinear', align_corners=False)
    preds_label = preds.max(dim=1).indices
    preds_label = preds_label.detach().cpu().numpy()
    labels = labels.detach().cpu().numpy()
    for pred, gt in zip(preds_label, labels):
        #print(pred.shape)
        #print(labels.shape)
        evaluator.update(pred, gt)
print('iou=', evaluator.overall_iou)
