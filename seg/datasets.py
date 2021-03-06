import torch
from torch.utils.data import Dataset
from pointnet.prepare_data import get_data_files
import numpy as np
from PIL import Image
import torchvision.transforms as T
from parse import parse
from torchvision.models.segmentation import fcn_resnet50, deeplabv3_resnet101
from torchvision.models import resnet18
import torch.nn.functional as F

from utils.file_utils import training_data_dir, testing_data_dir, data_root_dir, \
    training_image_feature_dir, testing_image_feature_dir, testing_data_perception_dir
from seg.seg_utils import extract_feature_from_backbone

class RGBImageDataset(Dataset):
    def __init__(self, data_path):
        super(RGBImageDataset, self).__init__()

        rgb_files, depth_files, label_files, meta_files = get_data_files(data_path,
                                                                         target_levels=(1, 2))
        self.rgb_files = rgb_files
        self.label_files = label_files
        self.transforms = T.Compose([T.Resize((224,224)),
                                     T.ToTensor(),
                                     T.Normalize(mean=[0.485, 0.456, 0.406],
                                                 std=[0.229, 0.224, 0.225])])

    def __len__(self):
        return len(self.rgb_files)

    def __getitem__(self, idx):
        rgb = Image.open(self.rgb_files[idx])
        rgb = self.transforms(rgb).float()
        #label = np.asarray(Image.open(self.label_files[idx]))
        #label = torch.from_numpy(label).long()
        return rgb, self.rgb_files[idx]

class RGBTrainingDataset(Dataset):
    def __init__(self, data_path, im_size, subset=False):
        super(RGBTrainingDataset, self).__init__()
        self.subset = subset
        rgb_files, depth_files, label_files, meta_files = get_data_files(data_path,
                                                                         target_levels=(1, 2))
        self.rgb_files = rgb_files
        self.label_files = label_files
        self.transforms_rgb = T.Compose([T.Resize(im_size),
                                         T.ToTensor(),
                                         T.Normalize(mean=[0.485, 0.456, 0.406],
                                                     std=[0.229, 0.224, 0.225])])
        self.transforms_label= T.Compose([T.Resize(im_size, interpolation=Image.NEAREST),
                                         T.ToTensor()])

    def __len__(self):
        if self.subset:
            return 1000
        return len(self.rgb_files)

    def __getitem__(self, idx):
        rgb = Image.open(self.rgb_files[idx])
        rgb = self.transforms_rgb(rgb).float()
        label = Image.open(self.label_files[idx])
        label = (self.transforms_label(label)*255).long().squeeze(dim=0)
        return rgb, label, self.rgb_files[idx]


class RGBTestingDataset(Dataset):
    def __init__(self, data_path, im_size):
        super(RGBTestingDataset, self).__init__()
        rgb_files, depth_files, label_files, meta_files = get_data_files(data_path,
                                                                         target_levels=(1, 2))
        self.rgb_files = rgb_files
        self.transforms_rgb = T.Compose([T.Resize(im_size),
                                         T.ToTensor(),
                                         T.Normalize(mean=[0.485, 0.456, 0.406],
                                                     std=[0.229, 0.224, 0.225])])
        self.transforms_label= T.Compose([T.Resize(im_size, interpolation=Image.NEAREST),
                                         T.ToTensor()])
        self.instance_ids = [parse('{}/v2.2/{}_label_kinect.png', fname)[-1] for fname in label_files]

    def __len__(self):
        return len(self.rgb_files)

    def __getitem__(self, idx):
        rgb = Image.open(self.rgb_files[idx])
        rgb = self.transforms_rgb(rgb).float()
        return rgb, self.instance_ids[idx]


class FCNTrainingDataset(Dataset):
    def __init__(self, image_data_path, feature_data_path):
        super(FCNTrainingDataset, self).__init__()

        rgb_files, depth_files, label_files, meta_files = get_data_files(image_data_path,
                                                                         target_levels=(1, 2))
        self.label_files = label_files
        self.instance_ids = [parse('{}/v2.2/{}_label_kinect.png', fname)[-1] for fname in label_files]
        #print(instance_ids[:5])
        self.feature_files = [feature_data_path+instance_id+'_feature.npy' for instance_id in self.instance_ids]
        #exit()

    def __len__(self):
        #return 100
        return len(self.label_files)

    def __getitem__(self, idx):
        feature = np.load(self.feature_files[idx])
        #upsampled_feature = F.interpolate(feature, size=[224, 224], mode='bilinear', align_corners=False)
        #upsampled_feature = upsampled_feature.view([2048,224,224])
        label = np.array(Image.open(self.label_files[idx]))
        #label[label==80]=79
        #label[label==81]=79
        label = torch.from_numpy(label).long()
        #return upsampled_feature, label
        return feature, label, self.instance_ids[idx]

class FCNTestingDataset(Dataset):
    def __init__(self, image_data_path, feature_data_path, subset=False):
        super(FCNTestingDataset, self).__init__()
        self.subset=subset
        rgb_files, depth_files, label_files, meta_files = get_data_files(image_data_path,
                                                                         target_levels=(1, 2))
        #self.label_files = label_files
        self.instance_ids = [parse('{}/v2.2/{}_label_kinect.png', fname)[-1] for fname in label_files]
        #print(instance_ids[:5])
        self.feature_files = [feature_data_path+instance_id+'_feature.npy' for instance_id in self.instance_ids]
        #exit()

    def __len__(self):
        if self.subset:
            return 100
        return len(self.feature_files)

    def __getitem__(self, idx):
        feature = np.load(self.feature_files[idx])
        return feature, self.instance_ids[idx]


if __name__ == '__main__':
    device = torch.device('cuda:1')
    rgb_dataset = RGBImageDataset(testing_data_perception_dir)
    training_loader = torch.utils.data.DataLoader(rgb_dataset, batch_size=8, shuffle=True, num_workers=8)
    model = deeplabv3_resnet101(pretrained=True, progress=True)
    backbone=model.backbone
    #model = resnet18(pretrained=True, progress=True)
    #return_layers = {'layer4': 'out'}
    #from torchvision.models._utils import IntermediateLayerGetter
    #backbone = IntermediateLayerGetter(model, return_layers=return_layers)

    extract_feature_from_backbone(testing_image_feature_dir, backbone, training_loader, device)

