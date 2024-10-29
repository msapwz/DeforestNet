import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init

from mmcv.cnn import build_activation_layer, build_conv_layer, build_norm_layer
from mmengine.model import BaseModule

from math import sqrt
import math

import random

import warnings



def init_linear(linear):
    init.xavier_normal(linear.weight)
    linear.bias.data.zero_()


def init_conv(conv, glu=True):
    init.kaiming_normal(conv.weight)
    if conv.bias is not None:
        conv.bias.data.zero_()


class EqualLR:
    def __init__(self, name):
        self.name = name

    def compute_weight(self, module):
        weight = getattr(module, self.name + '_orig')
        fan_in = weight.data.size(1) * weight.data[0][0].numel()

        return weight * sqrt(2 / fan_in)

    @staticmethod
    def apply(module, name):
        fn = EqualLR(name)

        weight = getattr(module, name)
        del module._parameters[name]
        module.register_parameter(name + '_orig', nn.Parameter(weight.data))
        module.register_forward_pre_hook(fn)

        return fn

    def __call__(self, module, input):
        weight = self.compute_weight(module)
        setattr(module, self.name, weight)


def equal_lr(module, name='weight'):
    EqualLR.apply(module, name)

    return module

# class CrossTemporalAttention(nn.Module):
#     def __init__(self, feature_dim):
#         super(CrossTemporalAttention, self).init()
#         self.query_transform = nn.Linear(feature_dim, feature_dim)
#         self.key_transform = nn.Linear(feature_dim, feature_dim)
#         self.value_transform = nn.Linear(feature_dim, feature_dim)
#         self.softmax = nn.Softmax(dim=-1)

#     def forward(self, x1, x2):
#         q1 = self.query_transform(x1)
#         k1 = self.key_transform(x1)
#         v1 = self.value_transform(x1)

#         q2 = self.query_transform(x2)
#         k2 = self.key_transform(x2)
#         v2 = self.value_transform(x2)

#         attn_weights_1 = self.softmax(torch.bmm(q1, k2.transpose(1, 2)))
#         attn_weights_2 = self.softmax(torch.bmm(q2, k1.transpose(1, 2)))

#         x1_out = torch.bmm(attn_weights_1, v1) 
#         x2_out = torch.bmm(attn_weights_2, v2) 

#         return x1_out, x2_out

class CrossTemporalAttention(nn.Module):
    def __init__(self, feature_dim):
        super(CrossTemporalAttention, self).__init__()
        self.query_transform = nn.Linear(feature_dim, feature_dim)
        self.key_transform = nn.Linear(feature_dim, feature_dim)
        self.value_transform = nn.Linear(feature_dim, feature_dim)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x1, x2):
        batch_size, channels, height, width = x1.size()  

        x1 = x1.view(batch_size, channels, -1)  
        x2 = x2.view(batch_size, channels, -1)  

        q1 = self.query_transform(x1.transpose(1, 2))  
        k1 = self.key_transform(x1.transpose(1, 2))    
        v1 = self.value_transform(x1.transpose(1, 2))  

        q2 = self.query_transform(x2.transpose(1, 2))  
        k2 = self.key_transform(x2.transpose(1, 2))    
        v2 = self.value_transform(x2.transpose(1, 2))  

        attn_weights_1 = self.softmax(torch.bmm(q1, k2.transpose(1, 2)))  
        attn_weights_2 = self.softmax(torch.bmm(q2, k1.transpose(1, 2)))  

        x1_out = torch.bmm(attn_weights_1, v1)  
        x2_out = torch.bmm(attn_weights_2, v2)  

        x1_out = x1_out.transpose(1, 2).view(batch_size, channels, height, width)
        x2_out = x2_out.transpose(1, 2).view(batch_size, channels, height, width)

        return x1_out, x2_out

#Difference module
def conv_diff(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.BatchNorm2d(out_channels),
        nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
        nn.ReLU()
    )

#Intermediate prediction module
def make_prediction(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.BatchNorm2d(out_channels),
        nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
    )

def resize(input,
            size=None,
            scale_factor=None,
            mode='nearest',
            align_corners=None,
            warning=True):
    if warning:
        if size is not None and align_corners:
            input_h, input_w = tuple(int(x) for x in input.shape[2:])
            output_h, output_w = tuple(int(x) for x in size)
            if output_h > input_h or output_w > output_h:
                if ((output_h > 1 and output_w > 1 and input_h > 1
                    and input_w > 1) and (output_h - 1) % (input_h - 1)
                        and (output_w - 1) % (input_w - 1)):
                    warnings.warn(
                        f'When align_corners={align_corners}, '
                        'the output would more aligned if '
                        f'input size {(input_h, input_w)} is `x+1` and '
                        f'out size {(output_h, output_w)} is `nx+1`')
    return F.interpolate(input, size, scale_factor, mode, align_corners)





class LearnableChannelExchange(nn.Module):
    def __init__(self, in_channels):
        super(LearnableChannelExchange, self).__init__()
        self.in_channels = in_channels
        self.channel_attention = nn.Linear(in_channels, in_channels, bias=False)
        self.softmax = nn.Softmax(dim=1)
    
    def forward(self, x1, x2):
        att_map1 = self.compute_attention(x1)
        att_map2 = self.compute_attention(x2)
        
        # Compute distances between attention maps
        distance = torch.norm(att_map1 - att_map2, p=2, dim=1, keepdim=True)
        # Generate exchange map
        exchange_map = distance < torch.median(distance, dim=1, keepdim=True)[0]
        
        N, C, H, W = x1.shape
        out_x1 = torch.where(exchange_map.expand_as(x1), x2, x1)
        out_x2 = torch.where(exchange_map.expand_as(x2), x1, x2)
        
        return out_x1, out_x2
    
    def compute_attention(self, x):
        # Compute attention for each channel
        att_map = self.channel_attention(x.mean(dim=[2, 3]))  # [N, C]
        att_map = att_map.view(x.size(0), -1, 1, 1)  # [N, C, 1, 1]
        return self.softmax(att_map)

# # class LearnableSpatialExchange(nn.Module):
# #     def __init__(self, in_channels):
# #         super(LearnableSpatialExchange, self).__init__()
# #         self.in_channels = in_channels
# #         self.spatial_attention = nn.Conv2d(in_channels, 1, kernel_size=1)
# #         self.softmax = nn.Softmax(dim=-1)
        
class LearnableSpatialExchange(nn.Module):
    def __init__(self, in_channels):
        super(LearnableSpatialExchange, self).__init__()
        self.in_channels = in_channels
        self.spatial_attention = nn.Conv2d(in_channels, 1, kernel_size=1)
    
    def forward(self, x1, x2):
        att_map1 = self.compute_attention(x1)  # [N, 1, H, W]
        att_map2 = self.compute_attention(x2)  # [N, 1, H, W]
        
        # Flatten spatial dimensions (H*W) for each attention map
        flat_att_map1 = att_map1.view(x1.size(0), -1)  # [N, H*W]
        flat_att_map2 = att_map2.view(x2.size(0), -1)  # [N, H*W]
        
        # Compute distances between attention maps
        distance = torch.norm(flat_att_map1 - flat_att_map2, p=2, dim=1, keepdim=True)
        
        # Generate spatial exchange map
        median_distance = torch.median(distance, dim=1, keepdim=True)[0]
        exchange_map = distance < median_distance
        
        # Reshape exchange map to match input spatial dimensions
        # exchange_map is [N, 1, 1, 1] after comparison; reshape it to [N, 1, H, W]
        exchange_map = exchange_map.view(x1.size(0), 1, 1, 1).expand(-1, 1, x1.size(2), x1.size(3))
        
        out_x1 = torch.where(exchange_map, x2, x1)
        out_x2 = torch.where(exchange_map, x1, x2)
        
        return out_x1, out_x2
    
    def compute_attention(self, x):
        # Compute attention for each spatial region
        att_map = self.spatial_attention(x)  # [N, 1, H, W]
        
        # Flatten and apply softmax over flattened spatial dimensions
        flat_att_map = att_map.view(x.size(0), -1)  # [N, H*W]
        att_map = F.softmax(flat_att_map, dim=1)  # Apply softmax over flattened dimension
        
        # Reshape back to original spatial dimensions
        att_map = att_map.view(x.size(0), 1, x.size(2), x.size(3))  # [N, 1, H, W]
        return att_map



class ChannelExchange(BaseModule):
    """
    channel exchange
    Args:
        p (float, optional): p of the features will be exchanged.
            Defaults to 1/2.
    """
    def __init__(self, p=1/2):
        super().__init__()
        assert p >= 0 and p <= 1
        self.p = int(1/p)

    def forward(self, x1, x2):
        N, c, h, w = x1.shape
        
        exchange_map = torch.arange(c) % self.p == 0
        exchange_mask = exchange_map.unsqueeze(0).expand((N, -1))

        out_x1, out_x2 = torch.zeros_like(x1), torch.zeros_like(x2)
        out_x1[~exchange_mask, ...] = x1[~exchange_mask, ...]
        out_x2[~exchange_mask, ...] = x2[~exchange_mask, ...]
        out_x1[exchange_mask, ...] = x2[exchange_mask, ...]
        out_x2[exchange_mask, ...] = x1[exchange_mask, ...]
        
        return out_x1, out_x2


class SpatialExchange(BaseModule):
    """
    spatial exchange
    Args:
        p (float, optional): p of the features will be exchanged.
            Defaults to 1/2.
    """
    def __init__(self, p=1/2):
        super().__init__()
        assert p >= 0 and p <= 1
        self.p = int(1/p)

    def forward(self, x1, x2):
        N, c, h, w = x1.shape
        exchange_mask = torch.arange(w) % self.p == 0

        out_x1, out_x2 = torch.zeros_like(x1), torch.zeros_like(x2)
        out_x1[..., ~exchange_mask] = x1[..., ~exchange_mask]
        out_x2[..., ~exchange_mask] = x2[..., ~exchange_mask]
        out_x1[..., exchange_mask] = x2[..., exchange_mask]
        out_x2[..., exchange_mask] = x1[..., exchange_mask]
        
        return out_x1, out_x2
    
    


class LearnableChannelExchange(nn.Module):
    def __init__(self, in_channels):
        super(LearnableChannelExchange, self).__init__()
        self.in_channels = in_channels
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.channel_attention = nn.Linear(in_channels, in_channels, bias=False,device=device)
        self.softmax = nn.Softmax(dim=1)
    
    def forward(self, x1, x2):
        att_map1 = self.compute_attention(x1)
        att_map2 = self.compute_attention(x2)
        
        distance = torch.norm(att_map1 - att_map2, p=2, dim=1, keepdim=True)
        exchange_map = distance < torch.median(distance, dim=1, keepdim=True)[0]
        
        N, C, H, W = x1.shape
        out_x1 = torch.where(exchange_map.expand_as(x1), x2, x1)
        out_x2 = torch.where(exchange_map.expand_as(x2), x1, x2)
        
        return out_x1, out_x2
    
    def compute_attention(self, x):
        # Compute attention for each channel
        att_map = self.channel_attention(x.mean(dim=[2, 3]))  
        att_map = att_map.view(x.size(0), -1, 1, 1) 
        return self.softmax(att_map)

        
class LearnableSpatialExchange(nn.Module):
    def __init__(self, in_channels):
        super(LearnableSpatialExchange, self).__init__()
        self.in_channels = in_channels
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.spatial_attention = nn.Conv2d(in_channels, 1, kernel_size=1,device=device)
    
    def forward(self, x1, x2):
        att_map1 = self.compute_attention(x1)  
        att_map2 = self.compute_attention(x2)  
        
        flat_att_map1 = att_map1.view(x1.size(0), -1)  
        flat_att_map2 = att_map2.view(x2.size(0), -1)  
        
        distance = torch.norm(flat_att_map1 - flat_att_map2, p=2, dim=1, keepdim=True)
    
        median_distance = torch.median(distance, dim=1, keepdim=True)[0]
        exchange_map = distance < median_distance
        
        exchange_map = exchange_map.view(x1.size(0), 1, 1, 1).expand(-1, 1, x1.size(2), x1.size(3))
        
        out_x1 = torch.where(exchange_map, x2, x1)
        out_x2 = torch.where(exchange_map, x1, x2)
        
        return out_x1, out_x2
    
    def compute_attention(self, x):
        att_map = self.spatial_attention(x)  # [N, 1, H, W]
        
        flat_att_map = att_map.view(x.size(0), -1)  
        att_map = F.softmax(flat_att_map, dim=1)
        
        att_map = att_map.view(x.size(0), 1, x.size(2), x.size(3))  
        return att_map


class LearnableChannelExchange2(nn.Module):
    def __init__(self, in_channels, init_threshold=0.5):
        super(LearnableChannelExchange2, self).__init__()
        self.in_channels = in_channels
        self.channel_attention = nn.Linear(
            in_channels, 
            in_channels, 
            bias=False, 
            )
        self.softmax = nn.Softmax(dim=1)
        
        self.threshold = nn.Parameter(
            torch.tensor(init_threshold), 
            requires_grad=True
        )
    
    def forward(self, x1, x2):
        att_map1 = self.compute_attention(x1)
        att_map2 = self.compute_attention(x2)
        
        distance = torch.norm(att_map1 - att_map2, p=2, dim=1, keepdim=True)
        
        median_distance = torch.median(distance, dim=1, keepdim=True)[0]
        exchange_map = torch.sigmoid(self.threshold * (distance - median_distance))

        out_x1 = exchange_map * x2 + (1 - exchange_map) * x1
        out_x2 = exchange_map * x1 + (1 - exchange_map) * x2
        
        return out_x1, out_x2

    
    def compute_attention(self, x):
        att_map = self.channel_attention(x.mean(dim=[2, 3]))  
        att_map = self.softmax(att_map)
        att_map = att_map.view(x.size(0), -1, 1, 1)  
        return att_map


class LearnableSpatialExchange2(nn.Module):
    def __init__(self, in_channels, init_threshold=0.5, init_scale=10.0):
        super(LearnableSpatialExchange2, self).__init__()
        self.in_channels = in_channels
        self.spatial_attention = nn.Conv2d(in_channels, 1, kernel_size=1)

        self.scale = nn.Parameter(torch.tensor(init_scale))
        self.threshold = nn.Parameter(torch.tensor(init_threshold), requires_grad=True)

    def forward(self, x1, x2):
        att_map1 = self.compute_attention(x1) 
        att_map2 = self.compute_attention(x2)  
        
        flat_att_map1 = att_map1.view(x1.size(0), -1) 
        flat_att_map2 = att_map2.view(x2.size(0), -1)  
        
        distance = torch.norm(flat_att_map1 - flat_att_map2, p=2, dim=1, keepdim=True)
        
        median_distance = torch.median(distance, dim=1, keepdim=True)[0]
        exchange_map = torch.sigmoid(self.threshold * (distance - median_distance))
        exchange_map = exchange_map.view(x1.size(0), 1, 1, 1).expand(-1, 1, x1.size(2), x1.size(3))
        
        out_x1 = exchange_map * x2 + (1 - exchange_map) * x1
        out_x2 = exchange_map * x1 + (1 - exchange_map) * x2
        
        return out_x1, out_x2

    def compute_attention(self, x):
        att_map = self.spatial_attention(x)  
        
        flat_att_map = att_map.view(x.size(0), -1)  
        att_map = F.softmax(self.scale * flat_att_map, dim=1)
        
        att_map = att_map.view(x.size(0), 1, x.size(2), x.size(3))  
        return att_map