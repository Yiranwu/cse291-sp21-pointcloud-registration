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
    training_image_feature_dir, testing_image_feature_dir, testing_data_perception_dir, load_pickle
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
        self.instance_ids = [parse('{}/v2.2/{}_label_kinect.png', fname)[-1] for fname in label_files]

    def __len__(self):
        if self.subset:
            return 1000
        return len(self.rgb_files)

    def __getitem__(self, idx):
        rgb = Image.open(self.rgb_files[idx])
        rgb = self.transforms_rgb(rgb).float()
        label = Image.open(self.label_files[idx])
        label = (self.transforms_label(label)*255).long().squeeze(dim=0)
        return rgb, label, self.instance_ids[idx]

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

class PCTrainingDataset(Dataset):
    def __init__(self, data_path, im_size, pc_im_size, subset=False):
        super(PCTrainingDataset, self).__init__()
        self.im_size = im_size
        self.subset = subset
        rgb_files, depth_files, label_files, meta_files = get_data_files(data_path,
                                                                         target_levels=(1, 2))
        self.rgb_files = rgb_files
        self.label_files = label_files
        self.depth_files = depth_files
        self.meta_files = meta_files
        self.transforms_rgb = T.Compose([T.Resize(im_size),
                                         T.ToTensor(),
                                         T.Normalize(mean=[0.485, 0.456, 0.406],
                                                     std=[0.229, 0.224, 0.225])])
        self.transforms_rgb_pc = T.Compose([T.Resize(pc_im_size),
                                         T.ToTensor(),
                                         T.Normalize(mean=[0.485, 0.456, 0.406],
                                                     std=[0.229, 0.224, 0.225])])
        self.transforms_depth = T.Compose([T.Resize(pc_im_size)])
        self.transforms_label= T.Compose([T.Resize(im_size, interpolation=Image.NEAREST),
                                         T.ToTensor()])
        self.instance_ids = [parse('{}/v2.2/{}_label_kinect.png', fname)[-1] for fname in label_files]

    def __len__(self):
        if self.subset:
            return 200
        return len(self.rgb_files)

    def __getitem__(self, idx):
        rgb = Image.open(self.rgb_files[idx])
        rgb= self.transforms_rgb(rgb).float()
        rgb_pc = Image.open(self.rgb_files[idx])
        rgb_pc= self.transforms_rgb_pc(rgb_pc).float()

        depth = np.array(self.transforms_depth(Image.open(self.depth_files[idx]))) / 1000
        meta = load_pickle(self.meta_files[idx])

        intrinsic = meta["intrinsic"]
        z = depth
        v, u = np.indices(z.shape)
        uv1 = np.stack([u + 0.5, v + 0.5, np.ones_like(z)], axis=-1)
        points_viewer = uv1 @ np.linalg.inv(intrinsic).T * z[..., None]  # [H, W, 3]
        xyz = torch.from_numpy(points_viewer).float().permute(2,0,1)
        pc = torch.cat([rgb_pc.view([3,-1]), xyz.view([3,-1])], dim=0)
        pc = (pc - pc.mean(dim=0))/pc.std(dim=0)

        label = Image.open(self.label_files[idx])
        label = (self.transforms_label(label)*255).long().squeeze(dim=0)

        return pc, rgb, label, self.instance_ids[idx]


class PCTestingDataset(Dataset):
    def __init__(self, data_path, im_size, pc_im_size, subset=False):
        super(PCTestingDataset, self).__init__()
        self.im_size = im_size
        self.subset = subset
        rgb_files, depth_files, label_files, meta_files = get_data_files(data_path,
                                                                         target_levels=(1, 2))
        self.rgb_files = rgb_files
        self.label_files = label_files
        self.depth_files = depth_files
        self.meta_files = meta_files
        self.transforms_rgb = T.Compose([T.Resize(im_size),
                                         T.ToTensor(),
                                         T.Normalize(mean=[0.485, 0.456, 0.406],
                                                     std=[0.229, 0.224, 0.225])])
        self.transforms_rgb_pc = T.Compose([T.Resize(pc_im_size),
                                         T.ToTensor(),
                                         T.Normalize(mean=[0.485, 0.456, 0.406],
                                                     std=[0.229, 0.224, 0.225])])
        self.transforms_depth = T.Compose([T.Resize(pc_im_size)])
        self.transforms_label= T.Compose([T.Resize(im_size, interpolation=Image.NEAREST),
                                         T.ToTensor()])
        self.instance_ids = [parse('{}/v2.2/{}_label_kinect.png', fname)[-1] for fname in label_files]

    def __len__(self):
        if self.subset:
            return 200
        return len(self.rgb_files)

    def __getitem__(self, idx):
        rgb = Image.open(self.rgb_files[idx])
        rgb= self.transforms_rgb(rgb).float()
        rgb_pc = Image.open(self.rgb_files[idx])
        rgb_pc= self.transforms_rgb_pc(rgb_pc).float()

        depth = np.array(self.transforms_depth(Image.open(self.depth_files[idx]))) / 1000
        meta = load_pickle(training_data_dir+'/2-27-8_meta.pkl')

        intrinsic = meta["intrinsic"]
        z = depth
        v, u = np.indices(z.shape)
        uv1 = np.stack([u + 0.5, v + 0.5, np.ones_like(z)], axis=-1)
        points_viewer = uv1 @ np.linalg.inv(intrinsic).T * z[..., None]  # [H, W, 3]
        xyz = torch.from_numpy(points_viewer).float().permute(2,0,1)
        pc = torch.cat([rgb_pc.view([3,-1]), xyz.view([3,-1])], dim=0)

        return pc, rgb, self.instance_ids[idx]

if __name__ == '__main__':
    device = torch.device('cuda:0')
    rgb_dataset = PCTrainingDataset(training_data_dir, 200)
    rgb, label, fname = rgb_dataset[0]
    print(rgb)


