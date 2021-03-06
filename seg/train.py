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

seg_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(seg_dir)
sys.path.append(root_dir)

from utils.file_utils import training_data_dir, training_image_feature_dir
from seg.datasets import FCNTrainingDataset
from seg.seg_utils import class_weights

lr=0.001
num_epochs=100
device = torch.device('cuda:0')

training_dataset = FCNTrainingDataset(training_data_dir, training_image_feature_dir)
training_loader = DataLoader(training_dataset, batch_size=8, shuffle=True, num_workers=1)
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
net.train()
#print('we have %d GPUs'%torch.cuda.device_count())
#net = torch.nn.DataParallel(net, device_ids=[0,1,2]).to(device)
net = net.to(device)
#net = net.cuda()

optimizer = torch.optim.Adam(net.parameters(), lr=lr)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5000, gamma=0.5)

loss_metric = torch.nn.CrossEntropyLoss(weight=class_weights)
loss_metric = loss_metric.to(device)
for epoch in range(num_epochs):
    for step, (data, label, _) in tqdm(enumerate(training_loader)):
        optimizer.zero_grad()
        data = data.to(device)
        label = label.to(device)
        pred = net(data)
        upsampled_pred=F.interpolate(pred, size=[720,1280], mode='bilinear', align_corners=False)
        #print(data.shape)
        #print(pred.shape)
        #print(upsampled_pred.shape)
        #exit()
        pred_label = upsampled_pred.max(dim=1).indices
        batch_size, h, w = label.shape
        is_object_tensor = (label<=78)
        is_correct_tensor=(pred_label==label)
        acc = (is_correct_tensor.float().sum())/(batch_size * h * w)
        object_acc = ((is_object_tensor.float() * is_correct_tensor.float()).sum()) / (is_object_tensor.float().sum())
        #print(upsampled_pred.shape)
        #print(label.shape)
        #exit()
        loss=loss_metric(upsampled_pred, label)
        loss.backward()
        optimizer.step()
        scheduler.step()
        print('loss=%f, acc=%f, object_acc=%f'%(loss.item(), acc.item(), object_acc.item()))
        if (step+1)%1000==0:
            torch.save(net.state_dict(), root_dir + '/saved_models/seg/deeplab_epoch%d_step%d.pth'%(epoch, (step+1)))
    #print('epoch %d, acc=%f'%(epoch, acc))


