from torch import nn
import torch
import math

from ..utils import *#ChannelExchange,SpatialExchange,LearnableSpatialExchange,LearnableChannelExchange

from mmcv.cnn.bricks import DropPath
from mmcv.cnn import build_norm_layer
from mmengine.model.weight_init import constant_init, normal_init, trunc_normal_init

from model.utils import CrossTemporalAttention

class MSCAModule(nn.Module):

    def __init__(self, d_embed: int):
        super().__init__()

        # 5x5 depth-wise convolution
        self.conv0 = nn.Conv2d(
            in_channels=d_embed,
            out_channels=d_embed,
            kernel_size=5,
            padding=2,
            groups=d_embed,
        )

        self.strip_conv_1_7 = nn.Conv2d(
            in_channels=d_embed,
            out_channels=d_embed,
            kernel_size=(1, 7),
            padding=(0, 3),
            groups=d_embed,
        )
        self.strip_conv_7_1 = nn.Conv2d(
            in_channels=d_embed,
            out_channels=d_embed,
            kernel_size=(7, 1),
            padding=(3, 0),
            groups=d_embed,
        )

        self.strip_conv_1_11 = nn.Conv2d(
            in_channels=d_embed,
            out_channels=d_embed,
            kernel_size=(1, 11),
            padding=(0, 5),
            groups=d_embed,
        )
        self.strip_conv_11_1 = nn.Conv2d(
            in_channels=d_embed,
            out_channels=d_embed,
            kernel_size=(11, 1),
            padding=(5, 0),
            groups=d_embed,
        )

        self.strip_conv_1_21 = nn.Conv2d(
            in_channels=d_embed,
            out_channels=d_embed,
            kernel_size=(1, 21),
            padding=(0, 10),
            groups=d_embed,
        )
        self.strip_conv_21_1 = nn.Conv2d(
            in_channels=d_embed,
            out_channels=d_embed,
            kernel_size=(21, 1),
            padding=(10, 0),
            groups=d_embed,
        )

        self.conv1 = nn.Conv2d(in_channels=d_embed, out_channels=d_embed, kernel_size=1)

    def forward(self, x):
        residual_x = x.clone()

        x = self.conv0(x)

        x_7 = self.strip_conv_1_7(x)
        x_7 = self.strip_conv_7_1(x_7)

        x_11 = self.strip_conv_1_11(x)
        x_11 = self.strip_conv_11_1(x_11)

        x_21 = self.strip_conv_1_21(x)
        x_21 = self.strip_conv_21_1(x_21)

        attn = self.conv1(x + x_7 + x_11 + x_21)

        return attn * residual_x
    


class AttentionBlock(nn.Module):

    def __init__(self, d_embed: int):
        super().__init__()
        self.conv0 = nn.Conv2d(in_channels=d_embed, out_channels=d_embed, kernel_size=1)
        self.activation = nn.GELU()
        self.msca = MSCAModule(d_embed=d_embed)
        self.conv1 = nn.Conv2d(in_channels=d_embed, out_channels=d_embed, kernel_size=1)

    def forward(self, x):
        residual_x = x.clone()

        x = self.conv0(x)
        x = self.activation(x)
        x = self.msca(x)
        x = self.conv1(x)

        return x + residual_x
    


class FFNBlock(nn.Module):

    def __init__(
        self,
        d_embed: int,
        expansion_ratio: int,
    ):
        super().__init__()
        hidden_features = d_embed * expansion_ratio

        self.conv0 = nn.Conv2d(
            in_channels=d_embed, out_channels=hidden_features, kernel_size=1
        )

        self.dwconv = nn.Conv2d(
            in_channels=hidden_features,
            out_channels=hidden_features,
            kernel_size=3,
            stride=1,
            padding=1,
            groups=hidden_features,
        )

        self.act = nn.GELU()

        self.conv1 = nn.Conv2d(
            in_channels=hidden_features, out_channels=d_embed, kernel_size=1
        )

    def forward(self, x):
        x = self.conv0(x)
        x = self.dwconv(x)
        x = self.act(x)
        x = self.conv1(x)
        return x
    


class MSCANBlock(nn.Module):

    def __init__(self, d_embed: int, expansion_ratio: int, drop_path_rate: float):
        super().__init__()
        _, self.norm0 = build_norm_layer(
            cfg=dict(type="SyncBN", requires_grad=True), num_features=d_embed
        )
        self.attention_block = MSCAModule(d_embed=d_embed)

        self.drop_path = (
            DropPath(drop_prob=drop_path_rate)
            if drop_path_rate > 0.0
            else nn.Identity()
        )

        _, self.norm1 = build_norm_layer(
            cfg=dict(type="SyncBN", requires_grad=True), num_features=d_embed
        )

        self.ffn_block = FFNBlock(d_embed=d_embed, expansion_ratio=expansion_ratio)

        layer_scale_init_value = 1e-2
        self.layer_scale_1 = nn.Parameter(
            data=layer_scale_init_value * torch.ones((d_embed)), requires_grad=True
        )
        self.layer_scale_2 = nn.Parameter(
            data=layer_scale_init_value * torch.ones((d_embed)), requires_grad=True
        )

    def forward(self, x, H, W):
        batch_size, patch_dim, embedding_dim = x.shape
        x = x.permute(0, 2, 1)
        x = x.view(batch_size, embedding_dim, H, W)

        residual_x = x
        x = self.attention_block(self.norm0(x))
        x = self.layer_scale_1.unsqueeze(-1).unsqueeze(-1) * x
        x = self.drop_path(x)
        x += residual_x

        residual_x = x
        x = self.ffn_block(self.norm1(x))
        x = self.layer_scale_2.unsqueeze(-1).unsqueeze(-1) * x
        x = self.drop_path(x)

        x += residual_x
        x = x.view(batch_size, embedding_dim, patch_dim)
        x = x.permute(0, 2, 1)

        return x

class StemConv(nn.Module):

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()

        self.conv0 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels // 2,
            kernel_size=3,
            stride=2,
            padding=1,
        )

        _, self.norm0 = build_norm_layer(
            cfg=dict(type="SyncBN", requires_grad=True), num_features=out_channels // 2
        )

        self.act = nn.GELU()

        self.conv1 = nn.Conv2d(
            in_channels=out_channels // 2,
            out_channels=out_channels,
            kernel_size=3,
            stride=2,
            padding=1,
        )

        _, self.norm1 = build_norm_layer(
            cfg=dict(type="SyncBN", requires_grad=True), num_features=out_channels
        )

    def forward(self, x):
        x = self.conv0(x)
        x = self.norm0(x)
        x = self.act(x)
        x = self.conv1(x)
        x = self.norm1(x)

        _, _, H, W = x.size()
        x = x.flatten(2)
        x = x.permute(0, 2, 1)

        return x, H, W

class OverlapPatchEmbed(nn.Module):

    def __init__(self, patch_size: int, stride: int, in_channels: int, d_embed: int):
        super().__init__()

        self.conv0 = nn.Conv2d(
            in_channels,
            d_embed,
            kernel_size=patch_size,
            stride=stride,
            padding=patch_size // 2,
        )
        _, self.norm0 = build_norm_layer(
            cfg=dict(type="SyncBN", requires_grad=True), num_features=d_embed
        )

    def forward(self, x):
        x = self.conv0(x)
        _, _, H, W = x.size()

        x = self.norm0(x)
        x = x.flatten(2)
        x = x.permute(0, 2, 1)

        return x, H, W
    


class MSCAN(nn.Module):

    def __init__(
        self,
        enc_config,
        # feat_ex = "N",
        ps=1/2,
        pc=1/2,
        init_spatial_threshold=0.5,   
        init_spatial_scale=10.0,
        init_channel_threshold=0.5,
    ):
        super(MSCAN, self).__init__()
        


        in_channels     = enc_config["in_channels"] 
        d_embeds        = enc_config["d_embeds"] 
        mlp_ratios      = enc_config["mlp_ratios"] 
        drop_path_rate  = enc_config["drop_path_rate"]
        depths          = enc_config["depths"]
        num_stages      = enc_config["num_stages"]
        exchange        = enc_config["feature_exchange"]
        cta             = enc_config["cta"] == "True"

        self.init_spatial_scale = init_spatial_scale
        self.init_channel_threshold = init_channel_threshold
        self.init_spatial_threshold = init_spatial_threshold
        self.ps = ps
        self.pc = pc

        super().__init__()

        self.num_stages = num_stages

        # If more than 4 stages, it will break, change the "get_feat_ex" to fix
        self.ps = ps
        self.pc = pc
        self.d_embeds = d_embeds
        # self.get_feat_ex()

        drop_path_rates = [
            x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))
        ]

        current_depth = 0

        for i in range(num_stages):
            if i == 0:
                patch_embedding = StemConv(in_channels, d_embeds[i])
            else:
                patch_embedding = OverlapPatchEmbed(3, 2, d_embeds[i - 1], d_embeds[i])

            mscan_blocks = nn.ModuleList(
                [
                    MSCANBlock(
                        d_embed=d_embeds[i],
                        expansion_ratio=mlp_ratios[i],
                        drop_path_rate=drop_path_rates[current_depth + L],
                    )
                    for L in range(depths[i])
                ]
            )

            layer_norm = nn.LayerNorm(normalized_shape=d_embeds[i])
            current_depth += depths[i]

            feature_exchange = self.get_feature_exchange(exchange[i],i)
            cta_ =  CrossTemporalAttention(d_embeds[i]) if cta else None
            if cta_ is not None:
                print("USING CTA")
            else:
                print("Not Using CTA")
            # feature_exchange = self.feature_exchange[i]

            setattr(self, f"patch_embed{i + 1}", patch_embedding)
            setattr(self, f"mscan_blocks{i + 1}", mscan_blocks)
            setattr(self, f"layer_norm{i + 1}", layer_norm)
            setattr(self, f"feature_exchange{i + 1}", feature_exchange)
            setattr(self, f"cta{i + 1}",cta_)

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                trunc_normal_init(m, std=0.02, bias=0.0)
            elif isinstance(m, nn.LayerNorm):
                constant_init(m, val=1.0, bias=0.0)
            elif isinstance(m, nn.Conv2d):
                fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                fan_out //= m.groups
                normal_init(m, mean=0, std=math.sqrt(2.0 / fan_out), bias=0)

    def get_feature_exchange(self,name,stage=None):
        if name=="N":
            print("No feature exchange")
            return None
        elif name=="se":
            print("SpatialExchange")
            return SpatialExchange(self.ps)
        elif name=="ce":
            print("ChannelExchange")
            return ChannelExchange(self.pc)
        elif name=="ls":
            print("LearnableSpatialExchange")
            return LearnableSpatialExchange(self.d_embeds[stage])
        elif name=="lc":
            print("LearnableChannelExchange")
            return LearnableChannelExchange(self.d_embeds[stage])
        elif name == "ls2":
            print("LearnableSpatialExchange2")
            return LearnableSpatialExchange2(
                self.d_embeds[stage],
                init_threshold=self.init_spatial_threshold,
                init_scale=self.init_spatial_scale
                )
        elif name == "lc2":
            print("LearnableChannelExchange2")
            return LearnableChannelExchange2(
                self.d_embeds[stage],
                init_threshold=self.init_channel_threshold
                )
        else:
            raise Exception("Provided learnable exchange does not exists.")
        
        

    # def forward(self, x1,x2):

    #     datas_in = [x1,x2]
    #     datas_out = []

    #     for x in datas_in:
    #         batch_size = x.shape[0]
    #         outs = []

    #         for i in range(self.num_stages):
    #             patch_embedding = getattr(self, f"patch_embed{i + 1}")
    #             mscan_blocks = getattr(self, f"mscan_blocks{i + 1}")
    #             layer_norm = getattr(self, f"layer_norm{i + 1}")
    #             feature_exchange = getattr(self, f"feature_exchange{i + 1}")

    #             x, H, W = patch_embedding(x)

    #             for mscan_block in mscan_blocks:
    #                 x = mscan_block(x, H, W)

    #             x = layer_norm(x)
    #             x = x.view(batch_size, H, W, -1)
    #             x = x.permute(0, 3, 1, 2).contiguous()
    #             if feature_exchange is not None:
    #                 x = feature_exchange(x)

    #             # if i > 0:
    #             #     outs.append(x)
    #             outs.append(x)
    #         datas_out.append(outs)

    #     return datas_out[0],datas_out[1]#outs


    def forward(self, x1,x2):

        # datas_in = [x1,x2]
        # datas_out = []
        X1_out = []
        X2_out = []

        batch_size = x1.shape[0]
        outs = []

        for i in range(self.num_stages):
            patch_embedding = getattr(self, f"patch_embed{i + 1}")
            mscan_blocks = getattr(self, f"mscan_blocks{i + 1}")
            layer_norm = getattr(self, f"layer_norm{i + 1}")
            feature_exchange = getattr(self, f"feature_exchange{i + 1}")
            cta = getattr(self, f"cta{i + 1}")

            x1, H1, W1 = patch_embedding(x1)
            x2, H2, W2 = patch_embedding(x2)

            for mscan_block in mscan_blocks:
                x1 = mscan_block(x1, H1, W1)
                x2 = mscan_block(x2, H2, W2)

            x1 = layer_norm(x1)
            x2 = layer_norm(x2)
            x1 = x1.view(batch_size, H1, W1, -1)
            x2 = x2.view(batch_size, H2, W2, -1)
            x1 = x1.permute(0, 3, 1, 2).contiguous()
            x2 = x2.permute(0, 3, 1, 2).contiguous()
            if feature_exchange is not None:
                x1,x2 = feature_exchange(x1,x2)

            if cta is not None:
                x1,x2 = cta(x1,x2)

            X1_out.append(x1)
            X2_out.append(x2)
        # datas_out.append(outs)
        return X1_out,X2_out
        # return datas_out[0],datas_out[1]#outs