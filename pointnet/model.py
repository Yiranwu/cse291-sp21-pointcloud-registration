# Code adapted from https://github.com/yanx27/Pointnet_Pointnet2_pytorch/blob/master/models/pointnet.py

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F

class FCNet(nn.Module):
    def __init__(self):
        super(FCNet, self).__init__()
        self.fc1 = nn.Linear(1024*3,1024)
        self.fc2=nn.Linear(1024,1024)
        self.fc3=nn.Linear(1024,1024)
        self.fc4=nn.Linear(1024,1024)
        self.fc5=nn.Linear(1024,16)
        self.fcRparam = nn.Linear(1024,6)
        self.fct = nn.Linear(1024,3)

    def forward(self, x):
        x=x.view([-1,1024*3])
        x=F.relu(self.fc1(x))
        x=F.relu(self.fc2(x))
        x=F.relu(self.fc3(x))
        x=F.relu(self.fc4(x))

        #x=self.fc5(x)
        #return x.view([-1,4,4])
        Rparam = self.fcRparam(x)
        t = self.fct(x)
        # print('t:',t[:2])
        R = network_output_to_R(Rparam)
        # R: b x 3 x 3, t: b x 3
        transformation = torch.cat([R, t.view([-1, 3, 1])], dim=2)
        pad_tensor = torch.tensor([0, 0, 0, 1], device=x.device, dtype=x.dtype).repeat((x.shape[0], 1))
        transformation = torch.cat([transformation, pad_tensor.view([-1, 1, 4])], dim=1)
        return transformation

class PointNetRot6d(nn.Module):
    def __init__(self, channel=3):
        super(PointNetRot6d, self).__init__()
        self.conv1 = torch.nn.Conv1d(channel, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        #self.bn1=nn.Identity()
        #self.bn2=nn.Identity()
        #self.bn3=nn.Identity()
        self.fc1 = nn.Linear(1024,512)
        self.bn_fc1=nn.BatchNorm1d(512)
        self.fc2 = nn.Linear(512,256)
        self.bn_fc2=nn.BatchNorm1d(256)
        self.fc_param = nn.Linear(256,9)
        self.fc_trivial=nn.Linear(256, 16)

    def forward(self, x):
        B, D, N = x.size()
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)
        x = F.relu(self.bn_fc1(self.fc1(x)))
        x = F.relu(self.bn_fc2(self.fc2(x)))
        #x=self.fc_trivial(x)
        #return x.view([-1,4,4])
        x=self.fc_param(x)
        Rparam = x[:,:6]
        t = x[:,6:]
        #print('t:',t[:2])
        R = network_output_to_R(Rparam)
        # R: b x 3 x 3, t: b x 3
        transformation = torch.cat([R, t.view([-1,3,1])], dim=2)
        pad_tensor = torch.tensor([0,0,0,1], device=x.device, dtype=x.dtype).repeat((B,1))
        transformation = torch.cat([transformation, pad_tensor.view([-1,1,4])], dim=1)
        return transformation



class PointNetRot6d_Wide(nn.Module):
    def __init__(self, channel=3):
        super(PointNetRot6d_Wide, self).__init__()
        self.conv1 = torch.nn.Conv1d(channel, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 64, 1)
        self.conv3 = torch.nn.Conv1d(64, 64, 1)
        self.conv4 = torch.nn.Conv1d(64, 128, 1)
        self.conv5 = torch.nn.Conv1d(128, 1024, 1)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(64)
        self.bn3 = nn.BatchNorm1d(64)
        self.bn4 = nn.BatchNorm1d(128)
        self.bn5 = nn.BatchNorm1d(1024)
        #self.bn1=nn.Identity()
        #self.bn2=nn.Identity()
        #self.bn3=nn.Identity()
        self.fc1 = nn.Linear(1024,1024)
        self.bn_fc1=nn.BatchNorm1d(1024)
        self.fc2 = nn.Linear(1024,512)
        self.bn_fc2=nn.BatchNorm1d(512)
        self.fc3 = nn.Linear(512,256)
        self.bn_fc3=nn.BatchNorm1d(256)
        self.fc_param = nn.Linear(256,9)
        self.fc_trivial=nn.Linear(256, 16)

    def forward(self, x):
        B, D, N = x.size()
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = self.bn5(self.conv5(x))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)
        x = F.relu(self.bn_fc1(self.fc1(x)))
        x = F.relu(self.bn_fc2(self.fc2(x)))
        x = F.relu(self.bn_fc3(self.fc3(x)))
        #x=self.fc_trivial(x)
        #return x.view([-1,4,4])
        x=self.fc_param(x)
        Rparam = x[:,:6]
        t = x[:,6:]
        #print('t:',t[:2])
        R = network_output_to_R(Rparam)
        # R: b x 3 x 3, t: b x 3
        transformation = torch.cat([R, t.view([-1,3,1])], dim=2)
        pad_tensor = torch.tensor([0,0,0,1], device=x.device, dtype=x.dtype).repeat((B,1))
        transformation = torch.cat([transformation, pad_tensor.view([-1,1,4])], dim=1)
        return transformation


class PointNetRot9d(nn.Module):
    def __init__(self, channel=3):
        super(PointNetRot9d, self).__init__()
        self.conv1 = torch.nn.Conv1d(channel, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        #self.bn1=nn.Identity()
        #self.bn2=nn.Identity()
        #self.bn3=nn.Identity()
        self.fc1 = nn.Linear(1024,512)
        self.bn_fc1=nn.BatchNorm1d(512)
        self.fc2 = nn.Linear(512,256)
        self.bn_fc2=nn.BatchNorm1d(256)
        self.fc_param = nn.Linear(256,12)
        self.fc_trivial=nn.Linear(256, 16)

    def forward(self, x):
        B, D, N = x.size()
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)
        x = F.relu(self.bn_fc1(self.fc1(x)))
        x = F.relu(self.bn_fc2(self.fc2(x)))
        #x=self.fc_trivial(x)
        #return x.view([-1,4,4])
        x=self.fc_param(x)
        Rparam = x[:,:9]
        t = x[:,9:]
        #print('t:',t[:2])
        R = symmetric_orthogonalization(Rparam)
        # R: b x 3 x 3, t: b x 3
        transformation = torch.cat([R, t.view([-1,3,1])], dim=2)
        pad_tensor = torch.tensor([0,0,0,1], device=x.device, dtype=x.dtype).repeat((B,1))
        transformation = torch.cat([transformation, pad_tensor.view([-1,1,4])], dim=1)
        return transformation

class PointNetfeat(nn.Module):
    def __init__(self, global_feat = True, feature_transform = False):
        super(PointNetfeat, self).__init__()
        self.conv1 = torch.nn.Conv1d(3, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.global_feat = global_feat
        self.feature_transform = feature_transform

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)
        return x

class PointNetCls(nn.Module):
    def __init__(self):
        super(PointNetCls, self).__init__()
        self.feat = PointNetfeat(global_feat=True)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 9)
        self.fc_trivial = nn.Linear(256,16)
        self.dropout = nn.Dropout(p=0.3)
        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.feat(x)
        x = F.relu(self.bn1(self.fc1(x)))
        x = F.relu(self.bn2(self.dropout(self.fc2(x))))
        xtrivial = self.fc_trivial(x)
        #print('xtrivial=', xtrivial[:3])
        x = self.fc3(x)
        return xtrivial.view([-1,4,4])


        Rparam = x[:,:6]
        t = x[:,6:]
        #print('t:',t[:2])
        R = network_output_to_R(Rparam)
        # R: b x 3 x 3, t: b x 3
        transformation = torch.cat([R, t.view([-1,3,1])], dim=2)
        pad_tensor = torch.tensor([0,0,0,1], device=x.device, dtype=x.dtype).repeat((x.shape[0],1))
        transformation = torch.cat([transformation, pad_tensor.view([-1,1,4])], dim=1)
        return transformation

def symmetric_orthogonalization(x):
  """Maps 9D input vectors onto SO(3) via symmetric orthogonalization.

  x: should have size [batch_size, 9]

  Output has size [batch_size, 3, 3], where each inner 3x3 matrix is in SO(3).
  """
  m = x.view(-1, 3, 3)
  #u, s, v = torch.svd(m)
  try:
      u, s, v = torch.svd(m)
  except:  # torch.svd may have convergence issues for GPU and CPU.
      u, s, v = torch.svd(m + 1e-4 * m.mean() * torch.rand(0,1))
  vt = torch.transpose(v, 1, 2)
  det = torch.det(torch.matmul(u, vt))
  det = det.view(-1, 1, 1)
  vt = torch.cat((vt[:, :2, :], vt[:, -1:, :] * det), 1)
  r = torch.matmul(u, vt)
  return r

def network_output_to_R_alternative(x):
    a1, a2 = x[..., :3], x[..., 3:]
    b1 = F.normalize(a1, dim=-1)
    b2 = a2 - (b1 * a2).sum(-1, keepdim=True) * b1
    b2 = F.normalize(b2, dim=-1)
    b3 = torch.cross(b1, b2, dim=-1)
    return torch.stack((b1, b2, b3), dim=-2)

def network_output_to_R(x):
    a1 = x[:,:3]
    #print('norm=',torch.norm(a1, p=2, dim=-1, keepdim=True))
    a1 = a1 / torch.norm(a1, p=2, dim=-1, keepdim=True)
    #print('a1 shape:',a1.shape)
    #print('a1 norm:', torch.norm(a1, p=2, dim=-1))
    #exit()
    a2 = x[:,3:]
    a2 = a2 / torch.norm(a2, p=2, dim=-1, keepdim=True)
    b1 = a1
    b2 = a2 - (b1*a2).sum(-1,keepdims=True) * b1
    b2 = b2 / torch.norm(b2, p=2, dim=-1, keepdim=True)
    b3 = torch.cross(b1,b2,dim=-1)
    b3 = b3 / torch.norm(b3, p=2, dim=-1, keepdim=True)
    R = torch.cat([b1.reshape([-1,3,1]),b2.reshape([-1,3,1]),b3.reshape([-1,3,1])],dim=-1)
    return R
