import torch
import torch.nn as nn
import torch.nn.functional as F
from seg.pointnet_utils import PointNetSetAbstractionMsg,PointNetFeaturePropagation


class PN2(nn.Module):
    def __init__(self, num_classes):
        super(PN2, self).__init__()

        self.sa1 = PointNetSetAbstractionMsg(1024, [0.05, 0.1], [16, 32], 6, [[16, 16, 32], [32, 32, 64]])
        self.sa2 = PointNetSetAbstractionMsg(256, [0.1, 0.2], [16, 32], 32+64, [[64, 64, 128], [64, 96, 128]])
        self.sa3 = PointNetSetAbstractionMsg(64, [0.2, 0.4], [16, 32], 128+128, [[128, 196, 256], [128, 196, 256]])
        self.sa4 = PointNetSetAbstractionMsg(16, [0.4, 0.8], [16, 32], 256+256, [[256, 256, 512], [256, 384, 512]])
        self.fp4 = PointNetFeaturePropagation(512+512+256+256, [256, 256])
        self.fp3 = PointNetFeaturePropagation(128+128+256, [256, 256])
        self.fp2 = PointNetFeaturePropagation(32+64+256, [256, 128])
        self.fp1 = PointNetFeaturePropagation(128, [128, 128, 128])

    def forward(self, xyz):
        l0_points = xyz
        l0_xyz = xyz[:,:3,:]

        l1_xyz, l1_points = self.sa1(l0_xyz, l0_points)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)
        l4_xyz, l4_points = self.sa4(l3_xyz, l3_points)

        l3_points = self.fp4(l3_xyz, l4_xyz, l3_points, l4_points)
        l2_points = self.fp3(l2_xyz, l3_xyz, l2_points, l3_points)
        l1_points = self.fp2(l1_xyz, l2_xyz, l1_points, l2_points)
        l0_points = self.fp1(l0_xyz, l1_xyz, None, l1_points)
        return l0_points
        # b x c x n

class PSPNet_pn2(torch.nn.Module):
    def __init__(self, n_classes, psp_model, im_size, pc_im_size, batch_norm=True, psp_out_feature=1024):
        super(PSPNet_pn2, self).__init__()
        self.features = psp_model.features

        for idx, m in reversed(list(enumerate(self.features.modules()))):
            if isinstance(m, nn.Conv2d):
                channels = m.out_channels
                break

        self.PSP = psp_model.PSP
        h_psp_out_feature = int(psp_out_feature / 2)
        q_psp_out_feature = int(psp_out_feature / 4)
        e_psp_out_feature = int(psp_out_feature / 8)
        self.upsampling1 = psp_model.upsampling1
        self.upsampling2 = psp_model.upsampling2
        self.upsampling3 = psp_model.upsampling3

        self.pn2 = PN2(num_classes=82)
        self.pc_im_size = pc_im_size

        self.classifier = nn.Sequential(nn.Conv2d(e_psp_out_feature+128, n_classes, kernel_size=1))

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.classifier.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x, pc):
        o = x
        for f in self.features:
            o = f(o)

        o = self.PSP(o)
        o = self.upsampling1(o)
        o = self.upsampling2(o)
        o = self.upsampling3(o)


        o = F.upsample(o, size=(x.shape[2], x.shape[3]), mode='bilinear')
        # [b, 128, 200, 355]
        feature_3d = self.pn2(pc)
        pc_im_h, pc_im_w = self.pc_im_size
        batch_size = feature_3d.shape[0]
        feature_3d = feature_3d.view([batch_size, -1, pc_im_h, pc_im_w])
        feature_3d = F.interpolate(feature_3d, size=[200, 355], mode='bilinear', align_corners=False)

        o=torch.cat([o, feature_3d], dim=1)

        o = self.classifier(o)

        return o

if __name__ == '__main__':
    import  torch
    model = get_model(13)
    xyz = torch.rand(6, 9, 8680)
    # b x c x n
    x = model(xyz)
    print(x.shape)