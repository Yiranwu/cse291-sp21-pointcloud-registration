import torch
import torch.nn as nn

class PSPNet_with_3d(torch.nn.Module):

    def __init__(self, n_classes):
        super(PSPNet_with_3d, self).__init__()
        self.features = FCN_2d.features
        self.copy_feature_info = FCN_2d.copy_feature_info
        self.score_pool3 = nn.Conv2d(self.copy_feature_info[-3].out_channels,
                                     n_classes, kernel_size=1)
        self.score_pool4 = nn.Conv2d(self.copy_feature_info[-2].out_channels,
                                     n_classes, kernel_size=1)

        self.upsampling2 = nn.ConvTranspose2d(n_classes, n_classes, kernel_size=4,
                                              stride=2, bias=False)
        self.upsampling8 = nn.ConvTranspose2d(n_classes, n_classes, kernel_size=16,
                                              stride=8, bias=False)

        for m in self.features.modules():
            if isinstance(m, nn.Conv2d):
                channels = m.out_channels

        self.classifier = nn.Sequential(nn.Conv2d(channels, n_classes, kernel_size=1), nn.Sigmoid())
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        saved_pools = []

        o = x
        for i in range(len(self.features)):
            o = self.features[i](o)
            if i == self.copy_feature_info[-3].index or\
                    i == self.copy_feature_info[-2].index:
                saved_pools.append(o)

        o = self.classifier(o)
        o = self.upsampling2(o)

        o2 = self.score_pool4(saved_pools[1])
        o = o[:, :, 1:1 + o2.size()[2], 1:1 + o2.size()[3]]
        o = o + o2

        o = self.upsampling2(o)

        o2 = self.score_pool3(saved_pools[0])
        o = o[:, :, 1:1 + o2.size()[2], 1:1 + o2.size()[3]]
        o = o + o2

        o = self.upsampling8(o)
        cx = int((o.shape[3] - x.shape[3]) / 2)
        cy = int((o.shape[2] - x.shape[2]) / 2)
        o = o[:, :, cy:cy + x.shape[2], cx:cx + x.shape[3]]

        return o