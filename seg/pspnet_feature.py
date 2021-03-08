import torch
import torch.nn as nn
import torch.nn.functional as F
from segmentation.models.pspnet import PSPModule, PSPUpsampling

class PSPFeatureNet(torch.nn.Module):
    def __init__(self, n_classes, psp_model, batch_norm=True, psp_out_feature=1024):
        super(PSPFeatureNet, self).__init__()
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

    def forward(self, x, feature_3d):
        o = x
        for f in self.features:
            o = f(o)

        o = self.PSP(o)
        o = self.upsampling1(o)
        o = self.upsampling2(o)
        o = self.upsampling3(o)


        o = F.upsample(o, size=(x.shape[2], x.shape[3]), mode='bilinear')
        o=torch.cat([o, feature_3d], dim=1)

        o = self.classifier(o)

        return o