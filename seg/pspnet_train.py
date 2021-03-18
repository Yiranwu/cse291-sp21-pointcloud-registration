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

model_name = "unet_resnet50"
device = torch.device('cuda:0')
batch_size = 8
n_classes = 82
num_epochs = 100
image_axis_minimum_size = 200
focal_gamma=2
pretrained = True
fixed_feature = False

training_dataset = RGBTrainingDataset(training_data_dir, image_axis_minimum_size)
training_loader = DataLoader(training_dataset, batch_size=batch_size, shuffle=True, num_workers=8)
### Model
net = all_models.model_from_name[model_name](n_classes, batch_size,
                                               pretrained=pretrained,
                                               fixed_feature=fixed_feature)
#net.load_state_dict(torch.load(root_dir + '/saved_models/seg/%s_epoch10_step1000.pth'%model_name))
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
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10000, gamma=0.5)

class_weights = torch.from_numpy(class_weights).float()
loss_metric = torch.nn.CrossEntropyLoss(weight=class_weights)
loss_metric = loss_metric.to(device)
for epoch in range(num_epochs):
    for step, (datas, labels, _) in tqdm(enumerate(training_loader)):
        evaluator = Evaluator()
        optimizer.zero_grad()
        datas = datas.to(device)
        labels = labels.to(device)
        preds = net(datas)
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
        focal_loss = (0.25 * (1 - pt) ** focal_gamma * ce_loss).mean()
        loss=focal_loss

        loss.backward()
        optimizer.step()
        preds_label = preds_label.detach().cpu().numpy()
        labels = labels.detach().cpu().numpy()
        for pred, gt in zip(preds_label, labels):
            evaluator.update(pred, gt)
        print('epoch=%d, loss=%f, iou=%f'%(epoch, loss.item(), evaluator.overall_iou))
        if (step+1)%1000==0:
            torch.save(net.state_dict(), root_dir + '/saved_models/seg/%s_focal%d_epoch%d_step%d.pth'%(model_name, focal_gamma, epoch, (step+1)))
    #print('epoch %d, acc=%f'%(epoch, acc))