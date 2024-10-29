import torch
from torch import nn
import torch.nn.functional as F
import numpy as np

# Losses from Change Former:
# https://github.com/wgcban/ChangeFormer/blob/main/models/losses.py

def cross_entropy(input, target, weight=None, reduction='mean',ignore_index=255):
    target = target.long()
    if target.dim() == 4:
        target = torch.squeeze(target, dim=1)
    if input.shape[-1] != target.shape[-1]:
        input = F.interpolate(input, 
                            size=target.shape[1:], 
                            mode='bilinear',
                            align_corners=True)

    return F.cross_entropy(input=input, 
                            target=target, 
                            weight=weight,
                            ignore_index=ignore_index, 
                            reduction=reduction
                            )

def get_alpha(supervised_loader):
    num_labels = 0
    for batch in supervised_loader:
        label_batch = batch['L']
        label_batch.data[label_batch.data==255] = 0 
        l_unique = torch.unique(label_batch.data)
        list_unique = [element.item() for element in l_unique.flatten()]
        num_labels = max(max(list_unique),num_labels)
    num_classes = num_labels + 1

    alpha = [0 for i in range(num_classes)]
    for batch in supervised_loader:
        label_batch = batch['L']
        label_batch.data[label_batch.data==255] = 0 
        l_unique = torch.unique(label_batch.data)
        list_unique = [element.item() for element in l_unique.flatten()]
        l_unique_count = torch.stack([(label_batch.data==x_u).sum() for x_u in l_unique]) # tensor([65920, 36480])
        list_count = [count.item() for count in l_unique_count.flatten()]
        for index in list_unique:
            alpha[index] += list_count[list_unique.index(index)]
    return alpha


def softmax_helper(x):
    rpt = [1 for _ in range(len(x.size()))]
    rpt[1] = x.size(1)
    x_max = x.max(1, keepdim=True)[0].repeat(*rpt)
    e_x = torch.exp(x - x_max)
    return e_x / e_x.sum(1, keepdim=True).repeat(*rpt)

class FocalLoss(nn.Module):

    def __init__(self, apply_nonlin=None, alpha=None, gamma=1, balance_index=0, smooth=1e-5, size_average=True):
        super(FocalLoss, self).__init__()
        self.apply_nonlin = apply_nonlin
        self.alpha = alpha
        self.gamma = gamma
        self.balance_index = balance_index
        self.smooth = smooth
        self.size_average = size_average

        if self.smooth is not None:
            if self.smooth < 0 or self.smooth > 1.0:
                raise ValueError('smooth value should be in [0,1]')

    def forward(self, logit, target):
        if self.apply_nonlin is not None:
            logit = self.apply_nonlin(logit)
        num_class = logit.shape[1]

        if logit.dim() > 2:
            logit = logit.view(logit.size(0), logit.size(1), -1)
            logit = logit.permute(0, 2, 1).contiguous()
            logit = logit.view(-1, logit.size(-1))
        target = torch.squeeze(target, 1)
        target = target.view(-1, 1)
	
        alpha = self.alpha

        if alpha is None:
            alpha = torch.ones(num_class, 1)
        elif isinstance(alpha, (list, np.ndarray)):
            assert len(alpha) == num_class
            alpha = torch.FloatTensor(alpha).view(num_class, 1)
            alpha = alpha / alpha.sum()
            alpha = 1/alpha 
        elif isinstance(alpha, float):
            alpha = torch.ones(num_class, 1)
            alpha = alpha * (1 - self.alpha)
            alpha[self.balance_index] = self.alpha

        else:
            raise TypeError('Not support alpha type')
        
        if alpha.device != logit.device:
            alpha = alpha.to(logit.device)

        idx = target.cpu().long()

        one_hot_key = torch.FloatTensor(target.size(0), num_class).zero_()
	
        idx[idx==225]=0
        
        one_hot_key = one_hot_key.scatter_(1, idx, 1)
        if one_hot_key.device != logit.device:
            one_hot_key = one_hot_key.to(logit.device)

        if self.smooth:
            one_hot_key = torch.clamp(
                one_hot_key, self.smooth/(num_class-1), 1.0 - self.smooth)
        pt = (one_hot_key * logit).sum(1) + self.smooth
        logpt = pt.log()

        gamma = self.gamma

        alpha = alpha[idx]
        alpha = torch.squeeze(alpha)
        loss = -1 * alpha * torch.pow((1 - pt), gamma) * logpt

        if self.size_average:
            loss = loss.mean()
        else:
            loss = loss.sum()
        return loss



from torch.autograd import Variable
def to_one_hot_var(tensor, nClasses, requires_grad=False):

    n, h, w = torch.squeeze(tensor, dim=1).size()
    one_hot = tensor.new(n, nClasses, h, w).fill_(0)
    one_hot = one_hot.scatter_(1, tensor.type(torch.int64).view(n, 1, h, w), 1)
    return Variable(one_hot, requires_grad=requires_grad)

class mIoULoss(nn.Module):
    def __init__(self, weight=None, size_average=True, n_classes=2):
        super(mIoULoss, self).__init__()
        self.classes = n_classes
        self.weights = Variable(weight)

    def forward(self, inputs, target, is_target_variable=False):

        N = inputs.size()[0]
        if is_target_variable:
            target_oneHot = to_one_hot_var(target.data, self.classes).float()
        else:
            target_oneHot = to_one_hot_var(target, self.classes).float()

        inputs = F.softmax(inputs, dim=1)


        inter = inputs * target_oneHot
        inter = inter.view(N, self.classes, -1).sum(2)

        union = inputs + target_oneHot - (inputs * target_oneHot)
        union = union.view(N, self.classes, -1).sum(2)

        loss = (self.weights * inter) / (union + 1e-8)

        return -torch.mean(loss)

class mmIoULoss(nn.Module):
    def __init__(self, n_classes=2):
        super(mmIoULoss, self).__init__()
        self.classes = n_classes

    def forward(self, inputs, target, is_target_variable=False):

        N = inputs.size()[0]
        if is_target_variable:
            target_oneHot = to_one_hot_var(target.data, self.classes).float()
        else:
            target_oneHot = to_one_hot_var(target, self.classes).float()

        inputs = F.softmax(inputs, dim=1)

        inter = inputs * target_oneHot
        inter = inter.view(N, self.classes, -1).sum(2)

        union = inputs + target_oneHot - (inputs * target_oneHot)
        union = union.view(N, self.classes, -1).sum(2)

        iou = inter/ (union + 1e-8)

        min_iou = torch.min(iou)

        loss = -min_iou-torch.mean(iou)
        return loss