from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F

from .FL import FTLoss
from .KD import DistillKL

class SemCKDLoss(nn.Module):
    """Cross-Layer Distillation with Semantic Calibration, AAAI2021"""
    def __init__(self, args):
        super(SemCKDLoss, self).__init__()
        self.criterion_kl = DistillKL(args.temp)
        self.criterion_fl = FTLoss()
        self.criterion_cls = nn.CrossEntropyLoss(reduction='none')
        self.crit = nn.MSELoss(reduction='none')
        self.alpha = args.alpha
        self.beta = args.beta

        
    def forward(self, s_value, f_target, s_pred, outputs, weight):
        bsz, num_stu = weight.shape
        ind_loss_kd = torch.zeros(bsz, num_stu).cuda()
        ind_loss_fl = torch.zeros(bsz, num_stu).cuda()
        ind_loss_cls = torch.zeros(bsz, num_stu).cuda()
        #one_matrix = torch.ones_like(weight).float().cuda()
        #temp_weight = torch.div(one_matrix, self.epsilon*one_matrix+weight)
        #weight = F.softmax(temp_weight, dim = 1)
        

        for i in range(num_stu):
            ind_loss_kd[:, i] = self.criterion_kl(s_value[i], f_target).reshape(bsz,-1).mean(-1)
            ind_loss_fl[:, i] = self.criterion_fl(s_value[i], f_target).reshape(bsz,-1).mean(-1)
            ind_loss_cls[:, i] = self.criterion_cls(s_pred[i], outputs)
            
        loss_kd = (weight * ind_loss_kd).sum()
        loss_fl = (weight * ind_loss_fl).sum()
        loss_cls = (weight * ind_loss_cls).sum()
        return loss_kd + self.alpha * loss_fl + self.beta * loss_cls