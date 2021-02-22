import torch
import numpy as np

class PoseLoss:
    def __init__(self, sym_Rs, is_zinfs, rot_axs):
        self.batch_manager = RLossBatchManager(sym_Rs, is_zinfs)
        self.rot_axs=rot_axs

    def get_skew_symmetric(self, v):
        # b x 3 x 1 -> b x 3 x 3
        b=v.shape[0]
        v1,v2,v3 = v.select(1,0), v.select(1,1),v.select(1,2)

        zero = torch.zeros([b,1], device=v.device)
        row0 = torch.stack([zero, -v3, v2],dim=1)
        row1 = torch.stack([v3, zero, -v1],dim=1)
        row2 = torch.stack([-v2, v1, zero],dim=1)
        skew = torch.stack([row0,row1,row2],dim=1)
        skew = skew.reshape([-1,3,3])
        return skew

    def get_R_to_align(self, pred, gt):
        b = pred.shape[0]
        c = (pred*gt).reshape([-1,3]).sum(-1).reshape([-1,1,1])
        v=torch.cross(pred.reshape([-1,3]), gt.reshape([-1,3]), dim=1).reshape([-1,3,1])
        skew = self.get_skew_symmetric(v)
        eye = torch.eye(3, device=pred.device, dtype=torch.float32).repeat((b,1,1))
        R = eye + skew + torch.matmul(skew,skew) / (1+c)
        return R

    def __call__(self, pred, gt, model_ids, with_symmetry=False):
        is_zinf = self.batch_manager.get_zinf(model_ids)
        #print('@loss.__call__: zinf pos=', np.where(is_zinf.cpu().numpy()))
        R = pred[:,:3,:3]
        t = pred[:,:3,3]
        R_gt = gt[:,:3,:3]
        t_gt = gt[:,:3,3]
        # R: b x 3 x 3
        # t: b x 3
        # is_zinf: b
        # model_id: b
        alpha=1.0
        t_loss = torch.nn.MSELoss()(t, t_gt)
        #R_loss=torch.nn.MSELoss()(R,R_gt)
        #return t_loss+R_loss
        #return t_loss + R_loss
        #R_loss=torch.nn.MSELoss(reduction='none')(R,R_gt).sum(-1).sum(-1).mean()
        #print('loss from MSE:',R_loss)
        if with_symmetry:
            #'''
            R_zinf = R[is_zinf]
            R_gt_zinf = R_gt[is_zinf]
            model_id_zinf = model_ids[is_zinf]
            R_znormal = R[is_zinf==0]
            R_gt_znormal = R_gt[is_zinf==0]
            model_id_znormal = model_ids[is_zinf==0]

            R_loss1=self.R_zinf_loss(R_zinf, R_gt_zinf, model_id_zinf)
            R_loss2=self.R_znormal_loss(R_znormal, R_gt_znormal, model_id_znormal, with_symmetry=True)
            R_loss_recovered = torch.zeros([pred.shape[0]],device=pred.device)
            R_loss_recovered[is_zinf] = R_loss1
            R_loss_recovered[is_zinf==0]=R_loss2
            #print('R_loss_tensor:',R_loss_recovered)

            R_loss = (R_loss1.sum() + R_loss2.sum())/pred.shape[0]
            #'''
            #R_loss = self.R_znormal_loss(R, R_gt, model_ids, with_symmetry=True)
            #R_loss= R_loss.sum() / pred.shape[0]
        else:
            R_loss = self.R_znormal_loss(R, R_gt, model_ids).sum()/pred.shape[0]
        #print('loss of mine:', R_loss)
        #print('R_loss_fro: ', R_loss_fro)
        #print("t_loss: %f, R_loss: %f"%(t_loss, R_loss))
        return alpha*t_loss+R_loss

    '''
    def geodesic_loss(self, R, R_gt, model_ids):
        # arccos(1/2(tr(R_gt R.T)-1))
        R_new, R_gt_new = self.batch_manager.batch(R,R_gt, model_ids)
        R_diff = R_new - R_gt_new
        R_diff_sq = R_diff.pow(2).sum(dim=-1).sum(dim=-1)
        loss = self.batch_manager.unbatch(R_diff_sq,model_ids)
        print("loss with sym=", loss)
        #print(loss)
        #loss=R_diff_sq
        return loss
    '''
    def R_znormal_loss(self,R, R_gt, model_ids, with_symmetry=False):
        if(R.shape[0]==0):
            return torch.zeros([0],device=R.device)
        if with_symmetry:
            R_new, R_gt_new = self.batch_manager.batch(R,R_gt, model_ids)
        else:
            R_new = R
            R_gt_new=R_gt
        # arccos(1/2(tr(R_gt R.T)-1))
        #R_new, R_gt_new = R, R_gt
        trace = torch.diagonal(torch.matmul(R_gt_new, R_new.transpose(1,2)), dim1=-2,dim2=-1).sum(-1)
        safe_trace = 0.5*(trace - 1) /(1+1e-6)
        #print('before acos:', safe_trace)
        acos = torch.acos(safe_trace)
        #print('after acos:', acos)
        #acos = (R_new-R_gt_new).pow(2).sum(dim=-1).sum(dim=-1)
        if with_symmetry:
            loss = self.batch_manager.unbatch(acos, model_ids)
        else:
            loss = acos
        #print('geodesic loss=',loss)
        #exit()
        #loss=R_diff
        #loss=self.batch_manager.unbatch(R_diff, model_ids)
        return loss

    def R_zinf_loss(self,R, R_gt, model_ids):
        if(R.shape[0]==0):
            return torch.zeros([0],device=R.device)
        R_new, R_gt_new = self.batch_manager.batch(R,R_gt,model_ids)
        rot_axis = self.rot_axs[model_ids].to(R.device).view([-1,3,1])
        rot_axis = self.batch_manager.expand_by_sym_fold(rot_axis, model_ids)
        pred_zaxis = torch.matmul(R_new, rot_axis)
        gt_zaxis = torch.matmul(R_gt_new, rot_axis)
        #print('pred_zaxis:', pred_zaxis)
        #print('gt_zaxis:', gt_zaxis)
        inner_product = (pred_zaxis * gt_zaxis).reshape([-1,3]).sum(dim=1)
        loss = self.batch_manager.unbatch(1-inner_product, model_ids)
        #print('loss:',loss)
        return loss


class RLossBatchManager:
    def __init__(self, sym_Rs, is_zinfs):
        # array, each element of shape [k, 3, 3]
        self.sym_Rs = sym_Rs
        self.is_zinfs = is_zinfs

    def get_zinf(self, model_ids):
        return self.is_zinfs[model_ids]

    def expand_by_sym_fold(self, data, model_ids):
        new_data = []
        repeat_shape = [1 for x in data.shape]
        for instance, model_id in zip(data, model_ids):
            sym_Rs = self.sym_Rs[model_id]
            sym_number = sym_Rs.shape[0]
            repeat_shape[0]=sym_number
            instance_duplicated = instance.repeat(repeat_shape)
            new_data.append(instance_duplicated)

        new_data = torch.cat(new_data, dim=0)
        return new_data

    def batch(self, Rs_pred, Rs_gt, model_ids):
        # Rs: [b, 3 ,3]
        # model_ids: [b]
        # return: two [k, 3, 3] tensors
        new_Rs_pred = []
        new_Rs_gt = []
        for R_pred, R_gt, model_id in zip(Rs_pred, Rs_gt, model_ids):
            sym_Rs = self.sym_Rs[model_id]
            sym_number = sym_Rs.shape[0]
            R_pred_duplicated = R_pred.repeat((sym_number,1,1))
            new_Rs_pred.append(R_pred_duplicated)
            R_gt_duplicated = R_gt.repeat((sym_number,1,1))
            R_gt_sym = torch.matmul(R_gt_duplicated, sym_Rs)
            #print("R_gt=", R_gt)
            #print("R_gt_sym=", R_gt_sym)
            new_Rs_gt.append(R_gt_sym)
        new_Rs_pred = torch.cat(new_Rs_pred, dim=0)
        new_Rs_gt = torch.cat(new_Rs_gt, dim=0)
        return new_Rs_pred, new_Rs_gt

    def unbatch(self, losses, model_ids):
        # losses: [k]
        # return: [b]
        index_sum=0
        loss_batch = []
        for model_id in model_ids:
            sym_number = self.sym_Rs[model_id].shape[0]
            model_loss = losses[index_sum:index_sum+sym_number]
            loss_batch.append(model_loss.min())
            index_sum+=sym_number
        loss_batch = torch.stack(loss_batch)
        return loss_batch
'''
s1 = torch.stack([torch.eye(3)])
s2 = torch.stack([torch.eye(3),torch.eye(3)])
s3 = torch.stack([torch.eye(3),torch.eye(3),torch.eye(3)])
sym_Rs = [s1,s2,s3]
manager = RLossBatchManager(sym_Rs)
Rs_pred = s3.clone()
Rs_gt = s3.clone()
model_ids = torch.tensor([0,1,2])
new_pred, new_gt = manager.batch(Rs_pred,Rs_gt, model_ids)
losses = torch.tensor([1,2,1,3,1,2])
print(manager.unbatch(losses, model_ids))
'''

'''
from pointnet import network_output_to_R
n=1000
x=torch.rand([2*n,6])
R = network_output_to_R(x)
Rpred = R[:n]
Rgt=R[n:]
ts = torch.rand([2*n,3])
tpred = ts[:n]
tgt = ts[n:]
is_zinf = torch.tensor(torch.cat([torch.zeros([n//2]), torch.ones([n//2])])).bool()
loss = pose_loss(Rpred, tpred ,Rgt, tgt,is_zinf)
print(loss.shape)
print(loss)
exit()
'''


