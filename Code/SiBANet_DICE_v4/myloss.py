import torch
import torch.nn as nn
from torch.nn import functional as F
import ipdb


class SoftDiceLoss(nn.Module):
    def __init__(self, ratio=None, weight=None, size_average=True):
        super(SoftDiceLoss, self).__init__()
        self.ratio = ratio

    def forward(self, logits, targets, ratio=None):
        smooth = 0.0000001
        num = targets.size(0)
        num_hns = int(self.ratio * num)

        # probs = F.sigmoid(logits)
        probs = logits
        m1 = probs.view(num, -1)
        m2 = targets.view(num, -1)
        intersection = (m1 * m2)

        score = 2. * (intersection.sum(1) + smooth) / (m1.sum(1) + m2.sum(1) + smooth)
        loss = 1 - score
        loss_hns, _ = loss.topk(num_hns)

        mloss_hns = loss_hns.sum() / num_hns

        return mloss_hns