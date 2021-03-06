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
from PIL import Image

seg_dir = os.path.dirname(__file__)
root_dir = os.path.dirname(seg_dir)
sys.path.append(root_dir)

from utils.file_utils import training_data_dir, training_image_feature_dir,\
    root_dir, data_root_dir, testing_data_perception_dir, testing_image_feature_dir
from seg.datasets import FCNTrainingDataset, FCNTestingDataset
from seg.seg_utils import class_weights

#os.system('rm '+ data_root_dir + "/testing_pred_perception/*.png")
#exit()
lr=0.001
num_epochs=100
device = torch.device('cuda:1')

#training_dataset = FCNTestingDataset(testing_data_perception_dir, testing_image_feature_dir)
training_dataset= FCNTestingDataset(training_data_dir, training_image_feature_dir, subset=True)
training_loader = DataLoader(training_dataset, batch_size=8, shuffle=False, num_workers=8)
'''
label_numbers = np.zeros([82])
for data, label in tqdm(training_loader):
    label_flatten = label.reshape([-1])
    unique, counts = np.unique(label_flatten, return_counts=True)
    for id, cnt in zip(unique, counts):
        label_numbers[id]=label_numbers[id]+cnt
label_weights = label_numbers/label_numbers.sum()
print(label_weights)
'''
class_weights = torch.from_numpy(class_weights).float()

net = DeepLabHead(2048, 82)
net.load_state_dict(torch.load(root_dir + '/saved_models/seg/deeplab_epoch1_step3000.pth'))
net.eval()
#print('we have %d GPUs'%torch.cuda.device_count())
#net = torch.nn.DataParallel(net, device_ids=[0,1,2]).to(device)
net = net.to(device)
for step, (data, instance_ids) in tqdm(enumerate(training_loader)):
    data = data.to(device)
    preds = net(data)
    upsampled_preds=F.interpolate(preds, size=[720,1280], mode='bilinear', align_corners=False)
    pred_labels = upsampled_preds.max(dim=1).indices
    pred_labels = pred_labels.cpu().numpy()
    for pred, instance_id in zip(pred_labels, instance_ids):
        pred = pred.astype(np.uint8)
        im = Image.fromarray(pred)
        im.save(data_root_dir + "/testing_pred_perception/%s_label_kinect.png"%instance_id)