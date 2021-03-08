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

from utils.file_utils import training_data_dir, training_image_feature_dir, data_root_dir
from seg.datasets import RGBTrainingDataset
from seg.seg_utils import class_weights
from benchmark_pose_and_detection.sem_seg_evaluator import Evaluator

model_name = "fcn8_resnet18"
device = torch.device('cuda:1')
batch_size = 16
n_classes = 82
num_epochs = 10
image_axis_minimum_size = 200
pretrained = True
fixed_feature = False

training_dataset = RGBTrainingDataset(training_data_dir, image_axis_minimum_size)
training_loader = DataLoader(training_dataset, batch_size=batch_size, shuffle=False, num_workers=8)
### Model
net = all_models.model_from_name[model_name](n_classes, batch_size,
                                               pretrained=pretrained,
                                               fixed_feature=fixed_feature)
net.load_state_dict(torch.load(root_dir + '/saved_models/seg/fcn_epoch20_step1000.pth'))
net.eval()
net.to(device)

from pathos.multiprocessing import ProcessingPool as Pool
pool = Pool(8)

def save_feature(feat, id):
    np.save(data_root_dir + '/fcn_feature/%s.npy'%id, feat)

for step, (datas, labels, instance_ids) in tqdm(enumerate(training_loader)):
    datas = datas.to(device)
    preds = net(datas)
    preds = preds.detach().cpu().numpy()
    pool.map(save_feature, preds, instance_ids)
    #for pred, instance_id in zip(preds, instance_ids):
    #    np.save(data_root_dir + '/fcn_feature/%s.npy'%instance_id, pred)
pool.close()
pool.join()