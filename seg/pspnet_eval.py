import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch.nn.functional as F
import numpy as np
from torchvision.models.segmentation import fcn_resnet50
from torchvision.models.segmentation.fcn import FCNHead
from torchvision.models.segmentation.deeplabv3 import DeepLabHead
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

model_name = "pspnet_resnet34"
device = torch.device('cuda:2')
batch_size = 4
n_classes = 82
num_epochs = 10
image_axis_minimum_size = 200
pretrained = True
fixed_feature = False

training_dataset = RGBTrainingDataset(training_data_dir, image_axis_minimum_size, subset=True)
training_loader = DataLoader(training_dataset, batch_size=batch_size, shuffle=False, num_workers=8)
### Model
net = all_models.model_from_name[model_name](n_classes, batch_size,
                                               pretrained=pretrained,
                                               fixed_feature=fixed_feature)
net.load_state_dict(torch.load(root_dir + '/saved_models/seg/pspnet_resnet18_epoch5_step1000.pth'))
net.train()
net.to(device)

evaluator = Evaluator()
for step, (datas, labels, _) in tqdm(enumerate(training_loader)):
    datas = datas.to(device)
    labels = labels.to(device)
    preds = net(datas)
    preds_label = preds.max(dim=1).indices
    preds_label = preds_label.detach().cpu().numpy()
    labels = labels.detach().cpu().numpy()
    for pred, gt in zip(preds_label, labels):
        evaluator.update(pred, gt)
print('iou=', evaluator.overall_iou)
