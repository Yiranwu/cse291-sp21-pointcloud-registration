import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch.nn.functional as F
import numpy as np
import os
import sys
from segmentation.models import all_models

seg_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(seg_dir)
sys.path.append(root_dir)

from utils.file_utils import training_data_dir, training_image_feature_dir
from seg.pn2_sem_seg_msg import PSPNet_pn2
from seg.pspnet_feature import PSPFeatureNet
from seg.datasets import PCTrainingDataset, RGBTrainingDataset
from seg.seg_utils import class_weights
from benchmark_pose_and_detection.sem_seg_evaluator import Evaluator

model_name = 'pspnet_resnet18'
device = torch.device('cuda:2')
batch_size = 4
n_classes = 82
num_epochs = 100
image_axis_minimum_size = 200
pretrained = True
fixed_feature = False

net_2d = all_models.model_from_name[model_name](n_classes, batch_size,
                                               pretrained=pretrained,
                                               fixed_feature=fixed_feature)
net_2d.load_state_dict(torch.load(root_dir + '/saved_models/seg/pspnet_resnet18_epoch10_step1000.pth'))
training_dataset = PCTrainingDataset(training_data_dir, 200, 100)
training_loader = DataLoader(training_dataset, batch_size=batch_size, shuffle=True, num_workers=8)
### Model
net = PSPNet_pn2(n_classes=82, psp_model = net_2d, im_size=[200,355],pc_im_size=[100,177])
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
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=500, gamma=0.3)

class_weights = torch.from_numpy(class_weights).float()
loss_metric = torch.nn.CrossEntropyLoss(weight=class_weights)
loss_metric = loss_metric.to(device)
for epoch in range(num_epochs):
    for step, (pcs, rgbs, labels, _) in tqdm(enumerate(training_loader)):
        evaluator = Evaluator()
        optimizer.zero_grad()
        #exit()
        pcs = pcs.to(device)
        rgbs=rgbs.to(device)
        labels = labels.to(device)

        preds = net(rgbs, pcs)
        preds_label = preds.max(dim=1).indices
        #_, h, w = label.shape
        #is_object_tensor = (label<=78)
        #is_correct_tensor=(pred_label==label)
        #acc = (is_correct_tensor.float().sum())/(batch_size * h * w)
        #object_acc = ((is_object_tensor.float() * is_correct_tensor.float()).sum()) / (is_object_tensor.float().sum())
        #print(upsampled_pred.shape)
        #print(label.shape)
        #exit()

        #loss=loss_metric(preds, labels)
        ce_loss = torch.nn.CrossEntropyLoss(reduction='none')(preds,labels)  # important to add reduction='none' to keep per-batch-item loss
        pt = torch.exp(-ce_loss)
        focal_loss = (0.25 * (1 - pt) ** 3 * ce_loss).mean()
        loss=focal_loss

        loss.backward()
        optimizer.step()
        preds_label = preds_label.detach().cpu().numpy()
        labels = labels.detach().cpu().numpy()
        for pred, gt in zip(preds_label, labels):
            evaluator.update(pred, gt)
        print('epoch=%d, loss=%f, iou=%f'%(epoch, loss.item(), evaluator.overall_iou))
        if (step+1)%200==0:
            torch.save(net.state_dict(), root_dir + '/saved_models/seg/psp_pn2_epoch%d_step%d.pth'%(epoch, (step+1)))
    #print('epoch %d, acc=%f'%(epoch, acc))