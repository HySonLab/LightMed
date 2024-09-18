import torch
import torch.nn as nn
class BCELoss(nn.Module):
    def __init__(self):
        super(BCELoss, self).__init__()
    def forward(self, pred, target):
        smooth = 1e-6
        size = pred.size(0)
        pred_ = pred.view(size, -1)
        target_ = target.view(size, -1)
        loss = - (target_ * torch.log(pred_ + smooth) + (1 - target_) * torch.log(1 - pred_ + smooth))
        return torch.mean(loss)


class DiceLoss(nn.Module):
    def __init__(self):
        super(DiceLoss, self).__init__()

    def forward(self, pred, target):
        smooth = 1e-6
        size = pred.size(0)

        pred_ = pred.view(size, -1)
        target_ = target.view(size, -1)
        intersection = pred_ * target_
        dice_score = (2 * intersection.sum(1) + smooth)/(pred_.sum(1) + target_.sum(1) + smooth)
        dice_loss = 1 - dice_score.sum()/size

        return dice_loss


class BceDiceLoss(nn.Module):
    def __init__(self, wb=1, wd=1):
        super(BceDiceLoss, self).__init__()
        self.dce = BCELoss()
        self.dice = DiceLoss()
        self.wb = wb
        self.wd = wd

    def forward(self, pred, target):
        bceloss = self.dce(pred, target)
        diceloss = self.dice(pred, target)

        loss = self.wd * diceloss + self.wb * bceloss
        return loss
    
