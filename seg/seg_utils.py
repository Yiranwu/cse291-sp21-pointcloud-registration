import numpy as np
from tqdm import tqdm
from parse import parse
import torch.nn.functional as F
from pointnet.prepare_data import get_data_files

label_weights = [1.14733435e-04, 6.12646032e-05, 5.29648618e-05, 1.93116274e-04,
                 1.29776476e-04, 7.44089084e-05, 7.24897823e-05, 0.00000000e+00,
                 2.15901412e-03, 0.00000000e+00, 0.00000000e+00, 1.49300221e-04,
                 7.29075340e-05, 1.76921477e-05, 3.45099917e-04, 0.00000000e+00,
                 1.70432310e-04, 9.15606463e-05, 7.63476562e-05, 1.96794208e-04,
                 2.74359538e-04, 6.40317564e-05, 3.48797201e-04, 0.00000000e+00,
                 7.58290835e-05, 1.94656395e-04, 2.35985695e-04, 0.00000000e+00,
                 2.42959482e-04, 2.32907850e-04, 1.35606314e-03, 6.27005073e-04,
                 2.50198161e-04, 2.95482133e-04, 2.13043168e-04, 1.40627903e-03,
                 0.00000000e+00, 2.46803431e-04, 0.00000000e+00, 4.31675257e-04,
                 7.61149541e-05, 0.00000000e+00, 1.39036160e-03, 1.59910663e-04,
                 0.00000000e+00, 2.31150626e-04, 0.00000000e+00, 0.00000000e+00,
                 7.83505543e-04, 5.36881330e-04, 1.41168258e-04, 5.52952429e-04,
                 6.22697257e-04, 0.00000000e+00, 0.00000000e+00, 7.14103371e-04,
                 6.73489719e-04, 2.18874286e-04, 1.34049131e-03, 2.33188341e-04,
                 1.92321506e-04, 3.30905490e-04, 4.81943902e-04, 1.71632261e-04,
                 2.77497152e-04, 2.61184263e-04, 1.10667589e-04, 1.32757749e-04,
                 1.16218759e-04, 4.00188531e-04, 3.73471318e-04, 1.71113598e-04,
                 4.43240469e-04, 2.52494077e-04, 1.16441967e-04, 1.04194530e-03,
                 1.27087900e-04, 0.00000000e+00, 9.70015914e-05, 2.67853367e-02,6.41161578e-01,3.08806102e-01]

label_weights = np.array(label_weights)
label_weights[label_weights==0]=1
class_weights = 1/label_weights

def extract_feature_from_backbone(save_dir, backbone, loader, device):
    backbone = backbone.to(device)
    backbone.eval()
    for data_batch, fnames in tqdm(loader):
        data_batch = data_batch.to(device)
        pred_batch = backbone(data_batch)['out'].detach().cpu().numpy()
        save_fnames = [parse('{}/v2.2/{}_color_kinect.png',fname)[-1] for fname in fnames]
        #print(save_fnames)
        #print(pred_batch.shape)
        #upsampled_batch = F.interpolate(pred_batch, size=[224,224], mode='bilinear', align_corners=False)
        #print(upsampled_batch.shape)
        #print(classifier(upsampled_batch).shape)
        #exit()
        for pred, fname in zip(pred_batch, save_fnames):
            np.save(save_dir+fname+'_feature.npy', pred)

if __name__ == '__main__':
    print(class_weights)
