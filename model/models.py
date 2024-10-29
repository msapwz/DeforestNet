from torch import nn
import torch

from model.backbones.segnext import MSCAN
from model.backbones.changeformer import *
from model.decode_heads.changeformer import *

from functools import partial


# 2 - SegNEXT + Changeformer decoder
MSCAN_CONFS = {
    "MSCAN_T": {
        "in_channels": 3,
        "d_embeds": [32, 64, 160, 256],
        "mlp_ratios": [8, 8, 4, 4],
        "drop_path_rate": 0.1,
        "depths": [3, 3, 5, 2],
        "num_stages": 4,
    },

    "MSCAN_S" : {
        "in_channels": 3,
        "d_embeds": [64, 128, 320, 512],
        "mlp_ratios": [8, 8, 4, 4],
        "drop_path_rate": 0.1,
        "depths": [2, 2, 4, 2],
        "num_stages": 4,
    },
    "MSCAN_B" : {
        "in_channels": 3,
        "d_embeds": [64, 128, 320, 512],
        "mlp_ratios": [8, 8, 4, 4],
        "drop_path_rate": 0.1,
        "depths": [3, 3, 12, 3],
        "num_stages": 4,
    },
    "MSCAN_L": {
        "in_channels": 3,
        "d_embeds": [64, 128, 320, 512],
        "mlp_ratios": [8, 8, 4, 4],
        "drop_path_rate": 0.1,
        "depths": [3, 5, 27, 3],
        "num_stages": 4,
    }
}

class DeforestationModel(nn.Module):

    def __init__(self,
                args,
                embed_dim=256,
                input_nc=3, 
                output_nc=2, 
                decoder_softmax=False,
                fusion = "concat",          #attention, average, sum
                encoder = "SegFormer",      #SegNext
                # att_type="att",             #cta
                use_cta="False",
                siamese_type="single",      #dual
                # feature_exchange="N",     #N,simple,learnable
                feature_exchange="N,N,N,N", #N,se,ce,ls,lc,ls2,lc2
                ):
        print(use_cta,"TYPE:",type(use_cta))
        super(DeforestationModel, self).__init__()
        self.siamese_type = args.siamese_type
        self.embedding_dim = embed_dim
        self.input_nc = input_nc
        self.output_nc = output_nc
        self.decoder_softmax = decoder_softmax
        self.mscan_v = args.mscan_v
        self.fusion = fusion
        self.embed_dim = 256
        self.init_spatial_scale = args.init_spatial_scale
        self.init_spatial_threshold = args.init_spatial_threshold
        self.init_channel_threshold = args.init_channel_threshold

        #TODO:
        self.encoder_name = encoder
        # self.att_type = att_type
        self.use_CTA = use_cta
        self.siamese_type = siamese_type
        self.feature_exchange = feature_exchange.split(",")

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.get_encoder(self.encoder_name)
        self.get_decoder()
        
    def get_encoder(self,name):
        if name=="SegFormer":
            print("ENCODER = SegFormer")
            self.embed_dims = [64, 128, 320, 512]
            self.depths     = [3, 3, 4, 3]
            self.drop_rate = 0.1
            self.attn_drop = 0.1
            self.drop_path_rate = 0.1

            self.in_channels = self.embed_dims

            self.encoder = EncoderSegFormer(
                img_size=256, 
                patch_size = 7, 
                in_chans=self.input_nc, 
                num_classes=self.output_nc, 
                embed_dims=self.embed_dims,
                num_heads = [1, 2, 4, 8], 
                mlp_ratios=[4, 4, 4, 4], 
                qkv_bias=True, 
                qk_scale=None, 
                drop_rate=self.drop_rate,
                attn_drop_rate = self.attn_drop, 
                drop_path_rate=self.drop_path_rate, 
                norm_layer=partial(nn.LayerNorm, eps=1e-6),
                depths=self.depths, 
                sr_ratios=[8, 4, 2, 1],
                # att_type=self.att_type,
                cta=self.use_CTA,
                feat_ex=self.feature_exchange,
                init_spatial_scale    =self.init_spatial_scale,
                init_spatial_threshold=self.init_spatial_threshold,
                init_channel_threshold=self.init_channel_threshold
                ).to(self.device)
            
        elif name=="SegNext":
            print("ENCODER = SegNext")
            mscan_v = MSCAN_CONFS[self.mscan_v]
            self.in_channels = mscan_v["d_embeds"]
            mscan_v["feature_exchange"] = self.feature_exchange
            mscan_v["cta"] = self.use_CTA
            self.encoder = MSCAN(mscan_v).to(self.device)

        else:
            raise TypeError(f"Provided encoder {name} do not exists.")

    def get_decoder(self):
        
        if self.siamese_type=="single":
            print("SINGLE DECODER")
            dec_func = Decoder
        elif self.siamese_type=="dual":
            print("DUAL DECODER")
            dec_func = DualDecoder
        else:
            raise TypeError(f"Provided decoder ({self.siamese_type}) do not exists")
    
        self.decoder = dec_func(
                input_transform='multiple_select', 
                in_index=[0, 1, 2, 3], 
                align_corners=False, 
                in_channels = self.in_channels,
                embedding_dim= self.embed_dim, 
                output_nc=self.output_nc, 
                decoder_softmax = False, 
                feature_strides=[2, 4, 8, 16],
                fusion=self.fusion
                )
        
    def forward(self,x1,x2):
        encoder_outs1,encoder_outs2 = self.encoder(x1,x2)

        if self.siamese_type=="single":
            out = self.decoder(encoder_outs1,encoder_outs2)

        elif self.siamese_type=="dual":

            out1 = self.decoder(encoder_outs1)
            out2 = self.decoder(encoder_outs2)
            fusion = self.fusion if self.fusion!="concat" else "average"
            out = []
            for i,j in zip(out1,out2):
                outp = fuse(fusion, i, j)
                out.append(outp)
            # out = fuse(self.fusion, out1, out2)

        return out#[-1]
    