from torch import nn
import torch
import torch.nn.functional as F

try:
    from model.utils import conv_diff,make_prediction
    from model.utils import resize
except ModuleNotFoundError:
    from utils import conv_diff,make_prediction
    from utils import resize

class TDec(nn.Module):
    """
    Transformer Decoder
    """
    def __init__(self, 
                input_transform='multiple_select', 
                in_index=[0, 1, 2, 3], 
                align_corners=True, 
                in_channels = [64, 128, 256, 512], 
                embedding_dim= 256, 
                output_nc=2, 
                decoder_softmax = False, 
                feature_strides=[4, 8, 16, 32]
                ):
        super(TDec, self).__init__()
        assert len(feature_strides) == len(in_channels)
        assert min(feature_strides) == feature_strides[0]
        self.feature_strides = feature_strides

        #input transforms
        self.input_transform = input_transform
        self.in_index = in_index
        self.align_corners = align_corners

        #MLP
        self.in_channels = in_channels
        self.embedding_dim = embedding_dim

        #Final prediction
        self.output_nc = output_nc

        c1_in_channels, c2_in_channels, c3_in_channels, c4_in_channels = self.in_channels

        self.linear_c4 = MLP(input_dim=c4_in_channels, embed_dim=self.embedding_dim)
        self.linear_c3 = MLP(input_dim=c3_in_channels, embed_dim=self.embedding_dim)
        self.linear_c2 = MLP(input_dim=c2_in_channels, embed_dim=self.embedding_dim)
        self.linear_c1 = MLP(input_dim=c1_in_channels, embed_dim=self.embedding_dim)

        self.linear_fuse = nn.Conv2d(   in_channels=self.embedding_dim*4, out_channels=self.embedding_dim,
                                        kernel_size=1)

        #self.linear_pred = nn.Conv2d(embedding_dim, self.num_classes, kernel_size=1)
        self.convd2x    = UpsampleConvLayer(self.embedding_dim, self.embedding_dim, kernel_size=4, stride=2)
        self.dense_2x   = nn.Sequential( ResidualBlock(self.embedding_dim))
        self.convd1x    = UpsampleConvLayer(self.embedding_dim, self.embedding_dim, kernel_size=4, stride=2)
        self.dense_1x   = nn.Sequential( ResidualBlock(self.embedding_dim))

        #Final prediction
        self.change_probability = ConvLayer(self.embedding_dim, self.output_nc, kernel_size=3, stride=1, padding=1)
        self.output_softmax     = decoder_softmax
        self.active             = nn.Softmax(dim=1) 

    def _transform_inputs(self, inputs):
        """Transform inputs for decoder.
        Args:
            inputs (list[Tensor]): List of multi-level img features.
        Returns:
            Tensor: The transformed inputs
        """

        if self.input_transform == 'resize_concat':
            inputs = [inputs[i] for i in self.in_index]
            upsampled_inputs = [
                resize(
                    input=x,
                    size=inputs[0].shape[2:],
                    mode='bilinear',
                    align_corners=self.align_corners) for x in inputs
            ]
            inputs = torch.cat(upsampled_inputs, dim=1)
        elif self.input_transform == 'multiple_select':
            inputs = [inputs[i] for i in self.in_index]
        else:
            inputs = inputs[self.in_index]

        return inputs

    def forward(self, inputs):
        x = self._transform_inputs(inputs)  # len=4, 1/4,1/8,1/16,1/32
        c1, c2, c3, c4 = x

        ############## MLP decoder on C1-C4 ###########
        n, _, h, w = c4.shape

        _c4 = self.linear_c4(c4).permute(0,2,1).reshape(n, -1, c4.shape[2], c4.shape[3])
        _c4 = resize(_c4, size=c1.size()[2:],mode='bilinear',align_corners=False)

        _c3 = self.linear_c3(c3).permute(0,2,1).reshape(n, -1, c3.shape[2], c3.shape[3])
        _c3 = resize(_c3, size=c1.size()[2:],mode='bilinear',align_corners=False)

        _c2 = self.linear_c2(c2).permute(0,2,1).reshape(n, -1, c2.shape[2], c2.shape[3])
        _c2 = resize(_c2, size=c1.size()[2:],mode='bilinear',align_corners=False)

        _c1 = self.linear_c1(c1).permute(0,2,1).reshape(n, -1, c1.shape[2], c1.shape[3])

        _c = self.linear_fuse(torch.cat([_c4, _c3, _c2, _c1], dim=1))

        x = self.convd2x(_c)
        x = self.dense_2x(x)
        x = self.convd1x(x)
        x = self.dense_1x(x)

        cp = self.change_probability(x)
        if self.output_softmax:
            cp = self.active(cp)

        return cp


class TDecV2(nn.Module):
    """
    Transformer Decoder
    """
    def __init__(self, input_transform='multiple_select', in_index=[0, 1, 2, 3], align_corners=True, 
                    in_channels = [64, 128, 256, 512], embedding_dim= 256, output_nc=2, 
                    decoder_softmax = False, feature_strides=[4, 8, 16, 32]):
        super(TDecV2, self).__init__()
        assert len(feature_strides) == len(in_channels)
        assert min(feature_strides) == feature_strides[0]
        self.feature_strides = feature_strides

        #input transforms
        self.input_transform = input_transform
        self.in_index = in_index
        self.align_corners = align_corners

        #MLP
        self.in_channels = in_channels
        self.embedding_dim = embedding_dim

        #Final prediction
        self.output_nc = output_nc

        c1_in_channels, c2_in_channels, c3_in_channels, c4_in_channels = self.in_channels

        self.linear_c4 = MLP(input_dim=c4_in_channels, embed_dim=self.embedding_dim)
        self.linear_c3 = MLP(input_dim=c3_in_channels, embed_dim=self.embedding_dim)
        self.linear_c2 = MLP(input_dim=c2_in_channels, embed_dim=self.embedding_dim)
        self.linear_c1 = MLP(input_dim=c1_in_channels, embed_dim=self.embedding_dim)

        self.linear_fuse = nn.Conv2d(   in_channels=self.embedding_dim*4, out_channels=self.embedding_dim,
                                        kernel_size=1)

        #self.linear_pred = nn.Conv2d(embedding_dim, self.num_classes, kernel_size=1)
        # self.convd2x    = UpsampleConvLayer(self.embedding_dim, self.embedding_dim, kernel_size=4, stride=2)
        # self.dense_2x   = nn.Sequential( ResidualBlock(self.embedding_dim))
        # self.convd1x    = UpsampleConvLayer(self.embedding_dim, self.embedding_dim, kernel_size=4, stride=2)
        # self.dense_1x   = nn.Sequential( ResidualBlock(self.embedding_dim))

        #Pixel Shiffle
        self.pix_shuffle_conv   = nn.Conv2d(in_channels=self.embedding_dim, out_channels=16*output_nc, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU()
        self.pix_shuffle        = nn.PixelShuffle(4)

        #Final prediction
        # self.change_probability = ConvLayer(self.embedding_dim, self.output_nc, kernel_size=3, stride=1, padding=1)
        
        #Final activation
        self.output_softmax     = decoder_softmax
        self.active             = nn.Softmax(dim=1) 

    def _transform_inputs(self, inputs):
        """Transform inputs for decoder.
        Args:
            inputs (list[Tensor]): List of multi-level img features.
        Returns:
            Tensor: The transformed inputs
        """

        if self.input_transform == 'resize_concat':
            inputs = [inputs[i] for i in self.in_index]
            upsampled_inputs = [
                resize(
                    input=x,
                    size=inputs[0].shape[2:],
                    mode='bilinear',
                    align_corners=self.align_corners) for x in inputs
            ]
            inputs = torch.cat(upsampled_inputs, dim=1)
        elif self.input_transform == 'multiple_select':
            inputs = [inputs[i] for i in self.in_index]
        else:
            inputs = inputs[self.in_index]

        return inputs

    def forward(self, inputs1, inputs2):
        x_1 = self._transform_inputs(inputs1)  # len=4, 1/4,1/8,1/16,1/32
        x_2 = self._transform_inputs(inputs2)  # len=4, 1/4,1/8,1/16,1/32

        c1_1, c2_1, c3_1, c4_1 = x_1
        c1_2, c2_2, c3_2, c4_2 = x_2

        ############## MLP decoder on C1-C4 ###########
        n, _, h, w = c4_1.shape

        _c4_1 = self.linear_c4(c4_1).permute(0,2,1).reshape(n, -1, c4_1.shape[2], c4_1.shape[3])
        _c4_1 = resize(_c4_1, size=c1_1.size()[2:],mode='bilinear',align_corners=False)
        _c4_2 = self.linear_c4(c4_2).permute(0,2,1).reshape(n, -1, c4_2.shape[2], c4_2.shape[3])
        _c4_2 = resize(_c4_2, size=c1_2.size()[2:],mode='bilinear',align_corners=False)

        _c3_1 = self.linear_c3(c3_1).permute(0,2,1).reshape(n, -1, c3_1.shape[2], c3_1.shape[3])
        _c3_1 = resize(_c3_1, size=c1_1.size()[2:],mode='bilinear',align_corners=False)
        _c3_2 = self.linear_c3(c3_2).permute(0,2,1).reshape(n, -1, c3_2.shape[2], c3_2.shape[3])
        _c3_2 = resize(_c3_2, size=c1_2.size()[2:],mode='bilinear',align_corners=False)

        _c2_1 = self.linear_c2(c2_1).permute(0,2,1).reshape(n, -1, c2_1.shape[2], c2_1.shape[3])
        _c2_1 = resize(_c2_1, size=c1_1.size()[2:],mode='bilinear',align_corners=False)
        _c2_2 = self.linear_c2(c2_2).permute(0,2,1).reshape(n, -1, c2_2.shape[2], c2_2.shape[3])
        _c2_2 = resize(_c2_2, size=c1_2.size()[2:],mode='bilinear',align_corners=False)

        _c1_1 = self.linear_c1(c1_1).permute(0,2,1).reshape(n, -1, c1_1.shape[2], c1_1.shape[3])
        _c1_2 = self.linear_c1(c1_2).permute(0,2,1).reshape(n, -1, c1_2.shape[2], c1_2.shape[3])

        _c = self.linear_fuse(torch.cat([torch.abs(_c4_1-_c4_2), torch.abs(_c3_1-_c3_2), torch.abs(_c2_1-_c2_2), torch.abs(_c1_1-_c1_2)], dim=1))

        # x = self.dense_2x(x)
        # x = self.convd1x(x)
        # x = self.dense_1x(x)

        # cp = self.change_probability(x)

        # cp = F.interpolate(_c, scale_factor=4, mode="nearest")
        x  = self.relu(self.pix_shuffle_conv(_c))
        cp = self.pix_shuffle(x)

        if self.output_softmax:
            cp = self.active(cp)

        return cp


# Transformer Decoder
class MLP(nn.Module):
    """
    Linear Embedding
    """
    def __init__(self, input_dim=2048, embed_dim=768):
        super().__init__()
        self.proj = nn.Linear(input_dim, embed_dim)

    def forward(self, x):
        x = x.flatten(2)
        x = x.transpose(1, 2)
        x = self.proj(x)
        return x
    
class ConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(ConvLayer, self).__init__()
#         reflection_padding = kernel_size // 2
#         self.reflection_pad = nn.ReflectionPad2d(reflection_padding)
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)

    def forward(self, x):
#         out = self.reflection_pad(x)
        out = self.conv2d(x)
        return out
    

class ResidualBlock(torch.nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = ConvLayer(channels, channels, kernel_size=3, stride=1, padding=1)
        self.conv2 = ConvLayer(channels, channels, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU()

    def forward(self, x):
        residual = x
        out = self.relu(self.conv1(x))
        out = self.conv2(out) * 0.1
        out = torch.add(out, residual)
        return out

class convprojection_base(nn.Module):
    def __init__(self, path=None, **kwargs):
        super(convprojection_base,self).__init__()

        # self.convd32x = UpsampleConvLayer(512, 512, kernel_size=4, stride=2)
        self.convd16x = UpsampleConvLayer(512, 320, kernel_size=4, stride=2)
        self.dense_4 = nn.Sequential(ResidualBlock(320))
        self.convd8x = UpsampleConvLayer(320, 128, kernel_size=4, stride=2)
        self.dense_3 = nn.Sequential(ResidualBlock(128))
        self.convd4x = UpsampleConvLayer(128, 64, kernel_size=4, stride=2)
        self.dense_2 = nn.Sequential(ResidualBlock(64))
        self.convd2x = UpsampleConvLayer(64, 16, kernel_size=4, stride=2)
        self.dense_1 = nn.Sequential( ResidualBlock(16))
        self.convd1x = UpsampleConvLayer(16, 8, kernel_size=4, stride=2)

    def forward(self,x1):

#         if x1[3].shape[3] != res32x.shape[3] and x1[3].shape[2] != res32x.shape[2]:
#             p2d = (0,-1,0,-1)
#             res32x = F.pad(res32x,p2d,"constant",0)
            
#         elif x1[3].shape[3] != res32x.shape[3] and x1[3].shape[2] == res32x.shape[2]:
#             p2d = (0,-1,0,0)
#             res32x = F.pad(res32x,p2d,"constant",0)
#         elif x1[3].shape[3] == res32x.shape[3] and x1[3].shape[2] != res32x.shape[2]:
#             p2d = (0,0,0,-1)
#             res32x = F.pad(res32x,p2d,"constant",0)

#         res16x = res32x + x1[3]
        res16x = self.convd16x(x1[3]) 

        if x1[2].shape[3] != res16x.shape[3] and x1[2].shape[2] != res16x.shape[2]:
            p2d = (0,-1,0,-1)
            res16x = F.pad(res16x,p2d,"constant",0)
        elif x1[2].shape[3] != res16x.shape[3] and x1[2].shape[2] == res16x.shape[2]:
            p2d = (0,-1,0,0)
            res16x = F.pad(res16x,p2d,"constant",0)
        elif x1[2].shape[3] == res16x.shape[3] and x1[2].shape[2] != res16x.shape[2]:
            p2d = (0,0,0,-1)
            res16x = F.pad(res16x,p2d,"constant",0)

        res8x = self.dense_4(res16x) + x1[2]
        res8x = self.convd8x(res8x) 
        res4x = self.dense_3(res8x) + x1[1]
        res4x = self.convd4x(res4x)
        res2x = self.dense_2(res4x) + x1[0]
        res2x = self.convd2x(res2x)
        x = res2x
        x = self.dense_1(x)
        x = self.convd1x(x)
        return x
    



class UpsampleConvLayer(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super(UpsampleConvLayer, self).__init__()
        self.conv2d = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=stride, padding=1)

    def forward(self, x):
        out = self.conv2d(x)
        return out

class DecoderTransformer_x2(nn.Module):
    """
    Transformer Decoder
    """
    def __init__(self, input_transform='multiple_select', in_index=[0, 1, 2, 3, 4], align_corners=True, 
                    in_channels = [32, 64, 128, 256, 512], embedding_dim= 64, output_nc=2, 
                    decoder_softmax = False, feature_strides=[2, 4, 8, 16, 32]):
        super(DecoderTransformer_x2, self).__init__()
        assert len(feature_strides) == len(in_channels)
        assert min(feature_strides) == feature_strides[0]
        self.feature_strides = feature_strides

        #input transforms
        self.input_transform = input_transform
        self.in_index = in_index
        self.align_corners = align_corners

        #MLP
        self.in_channels = in_channels
        self.embedding_dim = embedding_dim

        #Final prediction
        self.output_nc = output_nc

        c1_in_channels, c2_in_channels, c3_in_channels, c4_in_channels, c5_in_channels = self.in_channels

        self.linear_c5 = MLP(input_dim=c5_in_channels, embed_dim=self.embedding_dim)
        self.linear_c4 = MLP(input_dim=c4_in_channels, embed_dim=self.embedding_dim)
        self.linear_c3 = MLP(input_dim=c3_in_channels, embed_dim=self.embedding_dim)
        self.linear_c2 = MLP(input_dim=c2_in_channels, embed_dim=self.embedding_dim)
        self.linear_c1 = MLP(input_dim=c1_in_channels, embed_dim=self.embedding_dim)

        #Convolutional Difference Modules
        self.diff_c5   = conv_diff(in_channels=2*self.embedding_dim, out_channels=self.embedding_dim)
        self.diff_c4   = conv_diff(in_channels=3*self.embedding_dim, out_channels=self.embedding_dim)
        self.diff_c3   = conv_diff(in_channels=3*self.embedding_dim, out_channels=self.embedding_dim)
        self.diff_c2   = conv_diff(in_channels=3*self.embedding_dim, out_channels=self.embedding_dim)
        self.diff_c1   = conv_diff(in_channels=3*self.embedding_dim, out_channels=self.embedding_dim)

        #Taking outputs from middle of the encoder
        self.make_pred_c5 = make_prediction(in_channels=self.embedding_dim, out_channels=self.output_nc)
        self.make_pred_c4 = make_prediction(in_channels=self.embedding_dim, out_channels=self.output_nc)
        self.make_pred_c3 = make_prediction(in_channels=self.embedding_dim, out_channels=self.output_nc)
        self.make_pred_c2 = make_prediction(in_channels=self.embedding_dim, out_channels=self.output_nc)
        self.make_pred_c1 = make_prediction(in_channels=self.embedding_dim, out_channels=self.output_nc)

        self.linear_fuse = nn.Conv2d(   in_channels=self.embedding_dim*len(in_channels), out_channels=self.embedding_dim,
                                        kernel_size=1)

        #self.linear_pred = nn.Conv2d(embedding_dim, self.num_classes, kernel_size=1)
        self.convd2x    = UpsampleConvLayer(self.embedding_dim, self.embedding_dim, kernel_size=4, stride=2)
        self.dense_2x   = nn.Sequential( ResidualBlock(self.embedding_dim))
        self.convd1x    = UpsampleConvLayer(self.embedding_dim, self.embedding_dim, kernel_size=4, stride=2)
        self.dense_1x   = nn.Sequential( ResidualBlock(self.embedding_dim))

        #Final prediction
        self.change_probability = ConvLayer(self.embedding_dim, self.output_nc, kernel_size=3, stride=1, padding=1)
        
        #Final activation
        self.output_softmax     = decoder_softmax
        self.active             = nn.Sigmoid() 

    def _transform_inputs(self, inputs):
        """Transform inputs for decoder.
        Args:
            inputs (list[Tensor]): List of multi-level img features.
        Returns:
            Tensor: The transformed inputs
        """

        if self.input_transform == 'resize_concat':
            inputs = [inputs[i] for i in self.in_index]
            upsampled_inputs = [
                resize(
                    input=x,
                    size=inputs[0].shape[2:],
                    mode='bilinear',
                    align_corners=self.align_corners) for x in inputs
            ]
            inputs = torch.cat(upsampled_inputs, dim=1)
        elif self.input_transform == 'multiple_select':
            inputs = [inputs[i] for i in self.in_index]
        else:
            inputs = inputs[self.in_index]

        return inputs

    def forward(self, inputs1, inputs2):
        x_1 = self._transform_inputs(inputs1)  # len=4, 1/2,1/4,1/8,1/16,1/32
        x_2 = self._transform_inputs(inputs2)  # len=4, 1/2,1/4,1/8,1/16,1/32

        c1_1, c2_1, c3_1, c4_1, c5_1 = x_1
        c1_2, c2_2, c3_2, c4_2, c5_2 = x_2

        ############## MLP decoder on C1-C4 ###########
        n, _, h, w = c5_1.shape

        outputs = [] #Multi-scale outputs adding here
        
        _c5_1 = self.linear_c5(c5_1).permute(0,2,1).reshape(n, -1, c5_1.shape[2], c5_1.shape[3])
        _c5_2 = self.linear_c5(c5_2).permute(0,2,1).reshape(n, -1, c5_2.shape[2], c5_2.shape[3])
        _c5   = self.diff_c5(torch.cat((_c5_1, _c5_2), dim=1)) #Difference of features at x1/32 scale
        p_c5  = self.make_pred_c5(_c5) #Predicted change map at x1/32 scale
        outputs.append(p_c5) #x1/32 scale
        _c5_up= resize(_c5, size=c1_2.size()[2:], mode='bilinear', align_corners=False)

        _c4_1 = self.linear_c4(c4_1).permute(0,2,1).reshape(n, -1, c4_1.shape[2], c4_1.shape[3])
        _c4_2 = self.linear_c4(c4_2).permute(0,2,1).reshape(n, -1, c4_2.shape[2], c4_2.shape[3])
        _c4   = self.diff_c4(torch.cat((F.interpolate(_c5, scale_factor=2, mode="bilinear"), _c4_1, _c4_2), dim=1)) #Difference of features at x1/16 scale
        p_c4  = self.make_pred_c4(_c4) #Predicted change map at x1/16 scale
        outputs.append(p_c4) #x1/16 scale
        _c4_up= resize(_c4, size=c1_2.size()[2:], mode='bilinear', align_corners=False)

        _c3_1 = self.linear_c3(c3_1).permute(0,2,1).reshape(n, -1, c3_1.shape[2], c3_1.shape[3])
        _c3_2 = self.linear_c3(c3_2).permute(0,2,1).reshape(n, -1, c3_2.shape[2], c3_2.shape[3])
        _c3   = self.diff_c3(torch.cat((F.interpolate(_c4, scale_factor=2, mode="bilinear"), _c3_1, _c3_2), dim=1)) #Difference of features at x1/8 scale
        p_c3  = self.make_pred_c3(_c3) #Predicted change map at x1/8 scale
        outputs.append(p_c3) #x1/8 scale
        _c3_up= resize(_c3, size=c1_2.size()[2:], mode='bilinear', align_corners=False)

        _c2_1 = self.linear_c2(c2_1).permute(0,2,1).reshape(n, -1, c2_1.shape[2], c2_1.shape[3])
        _c2_2 = self.linear_c2(c2_2).permute(0,2,1).reshape(n, -1, c2_2.shape[2], c2_2.shape[3])
        _c2   = self.diff_c2(torch.cat((F.interpolate(_c3, scale_factor=2, mode="bilinear"), _c2_1, _c2_2), dim=1)) #Difference of features at x1/4 scale
        p_c2  = self.make_pred_c2(_c2) #Predicted change map at x1/4 scale
        outputs.append(p_c2) #x1/4 scale
        _c2_up= resize(_c2, size=c1_2.size()[2:], mode='bilinear', align_corners=False)

        _c1_1 = self.linear_c1(c1_1).permute(0,2,1).reshape(n, -1, c1_1.shape[2], c1_1.shape[3])
        _c1_2 = self.linear_c1(c1_2).permute(0,2,1).reshape(n, -1, c1_2.shape[2], c1_2.shape[3])
        _c1   = self.diff_c1(torch.cat((F.interpolate(_c2, scale_factor=2, mode="bilinear"), _c1_1, _c1_2), dim=1)) #Difference of features at x1/2 scale
        p_c1  = self.make_pred_c1(_c1) #Predicted change map at x1/2 scale
        outputs.append(p_c1) #x1/2 scale

        _c = self.linear_fuse(torch.cat((_c5_up, _c4_up, _c3_up, _c2_up, _c1), dim=1))

        x = self.convd2x(_c)
        x = self.dense_2x(x)
        cp = self.change_probability(x)
        outputs.append(cp)

        if self.output_softmax:
            temp = outputs
            outputs = []
            for pred in temp:
                outputs.append(self.active(pred))

        return outputs



class DecoderTransformer_x2(nn.Module):
    """
    Transformer Decoder
    """
    def __init__(self, input_transform='multiple_select', in_index=[0, 1, 2, 3, 4], align_corners=True, 
                    in_channels = [32, 64, 128, 256, 512], embedding_dim= 64, output_nc=2, 
                    decoder_softmax = False, feature_strides=[2, 4, 8, 16, 32]):
        super(DecoderTransformer_x2, self).__init__()
        assert len(feature_strides) == len(in_channels)
        assert min(feature_strides) == feature_strides[0]
        self.feature_strides = feature_strides

        #input transforms
        self.input_transform = input_transform
        self.in_index = in_index
        self.align_corners = align_corners

        #MLP
        self.in_channels = in_channels
        self.embedding_dim = embedding_dim

        #Final prediction
        self.output_nc = output_nc

        c1_in_channels, c2_in_channels, c3_in_channels, c4_in_channels, c5_in_channels = self.in_channels

        self.linear_c5 = MLP(input_dim=c5_in_channels, embed_dim=self.embedding_dim)
        self.linear_c4 = MLP(input_dim=c4_in_channels, embed_dim=self.embedding_dim)
        self.linear_c3 = MLP(input_dim=c3_in_channels, embed_dim=self.embedding_dim)
        self.linear_c2 = MLP(input_dim=c2_in_channels, embed_dim=self.embedding_dim)
        self.linear_c1 = MLP(input_dim=c1_in_channels, embed_dim=self.embedding_dim)

        #Convolutional Difference Modules
        self.diff_c5   = conv_diff(in_channels=2*self.embedding_dim, out_channels=self.embedding_dim)
        self.diff_c4   = conv_diff(in_channels=3*self.embedding_dim, out_channels=self.embedding_dim)
        self.diff_c3   = conv_diff(in_channels=3*self.embedding_dim, out_channels=self.embedding_dim)
        self.diff_c2   = conv_diff(in_channels=3*self.embedding_dim, out_channels=self.embedding_dim)
        self.diff_c1   = conv_diff(in_channels=3*self.embedding_dim, out_channels=self.embedding_dim)

        #Taking outputs from middle of the encoder
        self.make_pred_c5 = make_prediction(in_channels=self.embedding_dim, out_channels=self.output_nc)
        self.make_pred_c4 = make_prediction(in_channels=self.embedding_dim, out_channels=self.output_nc)
        self.make_pred_c3 = make_prediction(in_channels=self.embedding_dim, out_channels=self.output_nc)
        self.make_pred_c2 = make_prediction(in_channels=self.embedding_dim, out_channels=self.output_nc)
        self.make_pred_c1 = make_prediction(in_channels=self.embedding_dim, out_channels=self.output_nc)

        self.linear_fuse = nn.Conv2d(   in_channels=self.embedding_dim*len(in_channels), out_channels=self.embedding_dim,
                                        kernel_size=1)

        #self.linear_pred = nn.Conv2d(embedding_dim, self.num_classes, kernel_size=1)
        self.convd2x    = UpsampleConvLayer(self.embedding_dim, self.embedding_dim, kernel_size=4, stride=2)
        self.dense_2x   = nn.Sequential( ResidualBlock(self.embedding_dim))
        self.convd1x    = UpsampleConvLayer(self.embedding_dim, self.embedding_dim, kernel_size=4, stride=2)
        self.dense_1x   = nn.Sequential( ResidualBlock(self.embedding_dim))

        #Final prediction
        self.change_probability = ConvLayer(self.embedding_dim, self.output_nc, kernel_size=3, stride=1, padding=1)
        
        #Final activation
        self.output_softmax     = decoder_softmax
        self.active             = nn.Sigmoid() 

    def _transform_inputs(self, inputs):
        """Transform inputs for decoder.
        Args:
            inputs (list[Tensor]): List of multi-level img features.
        Returns:
            Tensor: The transformed inputs
        """

        if self.input_transform == 'resize_concat':
            inputs = [inputs[i] for i in self.in_index]
            upsampled_inputs = [
                resize(
                    input=x,
                    size=inputs[0].shape[2:],
                    mode='bilinear',
                    align_corners=self.align_corners) for x in inputs
            ]
            inputs = torch.cat(upsampled_inputs, dim=1)
        elif self.input_transform == 'multiple_select':
            inputs = [inputs[i] for i in self.in_index]
        else:
            inputs = inputs[self.in_index]

        return inputs

    def forward(self, inputs1, inputs2):
        x_1 = self._transform_inputs(inputs1)  # len=4, 1/2,1/4,1/8,1/16,1/32
        x_2 = self._transform_inputs(inputs2)  # len=4, 1/2,1/4,1/8,1/16,1/32

        c1_1, c2_1, c3_1, c4_1, c5_1 = x_1
        c1_2, c2_2, c3_2, c4_2, c5_2 = x_2

        ############## MLP decoder on C1-C4 ###########
        n, _, h, w = c5_1.shape

        outputs = [] #Multi-scale outputs adding here
        
        _c5_1 = self.linear_c5(c5_1).permute(0,2,1).reshape(n, -1, c5_1.shape[2], c5_1.shape[3])
        _c5_2 = self.linear_c5(c5_2).permute(0,2,1).reshape(n, -1, c5_2.shape[2], c5_2.shape[3])
        _c5   = self.diff_c5(torch.cat((_c5_1, _c5_2), dim=1)) #Difference of features at x1/32 scale
        p_c5  = self.make_pred_c5(_c5) #Predicted change map at x1/32 scale
        outputs.append(p_c5) #x1/32 scale
        _c5_up= resize(_c5, size=c1_2.size()[2:], mode='bilinear', align_corners=False)

        _c4_1 = self.linear_c4(c4_1).permute(0,2,1).reshape(n, -1, c4_1.shape[2], c4_1.shape[3])
        _c4_2 = self.linear_c4(c4_2).permute(0,2,1).reshape(n, -1, c4_2.shape[2], c4_2.shape[3])
        _c4   = self.diff_c4(torch.cat((F.interpolate(_c5, scale_factor=2, mode="bilinear"), _c4_1, _c4_2), dim=1)) #Difference of features at x1/16 scale
        p_c4  = self.make_pred_c4(_c4) #Predicted change map at x1/16 scale
        outputs.append(p_c4) #x1/16 scale
        _c4_up= resize(_c4, size=c1_2.size()[2:], mode='bilinear', align_corners=False)

        _c3_1 = self.linear_c3(c3_1).permute(0,2,1).reshape(n, -1, c3_1.shape[2], c3_1.shape[3])
        _c3_2 = self.linear_c3(c3_2).permute(0,2,1).reshape(n, -1, c3_2.shape[2], c3_2.shape[3])
        _c3   = self.diff_c3(torch.cat((F.interpolate(_c4, scale_factor=2, mode="bilinear"), _c3_1, _c3_2), dim=1)) #Difference of features at x1/8 scale
        p_c3  = self.make_pred_c3(_c3) #Predicted change map at x1/8 scale
        outputs.append(p_c3) #x1/8 scale
        _c3_up= resize(_c3, size=c1_2.size()[2:], mode='bilinear', align_corners=False)

        _c2_1 = self.linear_c2(c2_1).permute(0,2,1).reshape(n, -1, c2_1.shape[2], c2_1.shape[3])
        _c2_2 = self.linear_c2(c2_2).permute(0,2,1).reshape(n, -1, c2_2.shape[2], c2_2.shape[3])
        _c2   = self.diff_c2(torch.cat((F.interpolate(_c3, scale_factor=2, mode="bilinear"), _c2_1, _c2_2), dim=1)) #Difference of features at x1/4 scale
        p_c2  = self.make_pred_c2(_c2) #Predicted change map at x1/4 scale
        outputs.append(p_c2) #x1/4 scale
        _c2_up= resize(_c2, size=c1_2.size()[2:], mode='bilinear', align_corners=False)

        _c1_1 = self.linear_c1(c1_1).permute(0,2,1).reshape(n, -1, c1_1.shape[2], c1_1.shape[3])
        _c1_2 = self.linear_c1(c1_2).permute(0,2,1).reshape(n, -1, c1_2.shape[2], c1_2.shape[3])
        _c1   = self.diff_c1(torch.cat((F.interpolate(_c2, scale_factor=2, mode="bilinear"), _c1_1, _c1_2), dim=1)) #Difference of features at x1/2 scale
        p_c1  = self.make_pred_c1(_c1) #Predicted change map at x1/2 scale
        outputs.append(p_c1) #x1/2 scale

        _c = self.linear_fuse(torch.cat((_c5_up, _c4_up, _c3_up, _c2_up, _c1), dim=1))

        x = self.convd2x(_c)
        x = self.dense_2x(x)
        cp = self.change_probability(x)
        outputs.append(cp)

        if self.output_softmax:
            temp = outputs
            outputs = []
            for pred in temp:
                outputs.append(self.active(pred))

        return outputs




class DecoderTransformer_v3(nn.Module):
    """
    Transformer Decoder
    """
    def __init__(self, input_transform='multiple_select', in_index=[0, 1, 2, 3], align_corners=True, 
                    in_channels = [32, 64, 128, 256], embedding_dim= 64, output_nc=2, 
                    decoder_softmax = False, feature_strides=[2, 4, 8, 16]):
        super(DecoderTransformer_v3, self).__init__()
        #assert
        assert len(feature_strides) == len(in_channels)
        assert min(feature_strides) == feature_strides[0]
        
        #settings
        self.feature_strides = feature_strides
        self.input_transform = input_transform
        self.in_index        = in_index
        self.align_corners   = align_corners
        self.in_channels     = in_channels
        self.embedding_dim   = embedding_dim
        self.output_nc       = output_nc
        c1_in_channels, c2_in_channels, c3_in_channels, c4_in_channels = self.in_channels

        #MLP decoder heads
        self.linear_c4 = MLP(input_dim=c4_in_channels, embed_dim=self.embedding_dim)
        self.linear_c3 = MLP(input_dim=c3_in_channels, embed_dim=self.embedding_dim)
        self.linear_c2 = MLP(input_dim=c2_in_channels, embed_dim=self.embedding_dim)
        self.linear_c1 = MLP(input_dim=c1_in_channels, embed_dim=self.embedding_dim)

        #convolutional Difference Modules
        self.diff_c4   = conv_diff(in_channels=2*self.embedding_dim, out_channels=self.embedding_dim)
        self.diff_c3   = conv_diff(in_channels=2*self.embedding_dim, out_channels=self.embedding_dim)
        self.diff_c2   = conv_diff(in_channels=2*self.embedding_dim, out_channels=self.embedding_dim)
        self.diff_c1   = conv_diff(in_channels=2*self.embedding_dim, out_channels=self.embedding_dim)

        #taking outputs from middle of the encoder
        self.make_pred_c4 = make_prediction(in_channels=self.embedding_dim, out_channels=self.output_nc)
        self.make_pred_c3 = make_prediction(in_channels=self.embedding_dim, out_channels=self.output_nc)
        self.make_pred_c2 = make_prediction(in_channels=self.embedding_dim, out_channels=self.output_nc)
        self.make_pred_c1 = make_prediction(in_channels=self.embedding_dim, out_channels=self.output_nc)

        #Final linear fusion layer
        self.linear_fuse = nn.Sequential(
            nn.Conv2d(   in_channels=self.embedding_dim*len(in_channels), out_channels=self.embedding_dim,
                                        kernel_size=1),
            nn.BatchNorm2d(self.embedding_dim)
        )

        #Final predction head
        self.convd2x    = UpsampleConvLayer(self.embedding_dim, self.embedding_dim, kernel_size=4, stride=2)
        self.dense_2x   = nn.Sequential( ResidualBlock(self.embedding_dim))
        self.convd1x    = UpsampleConvLayer(self.embedding_dim, self.embedding_dim, kernel_size=4, stride=2)
        self.dense_1x   = nn.Sequential( ResidualBlock(self.embedding_dim))
        self.change_probability = ConvLayer(self.embedding_dim, self.output_nc, kernel_size=3, stride=1, padding=1)
        
        #Final activation
        self.output_softmax     = decoder_softmax
        self.active             = nn.Sigmoid() 

    def _transform_inputs(self, inputs):
        """Transform inputs for decoder.
        Args:
            inputs (list[Tensor]): List of multi-level img features.
        Returns:
            Tensor: The transformed inputs
        """

        if self.input_transform == 'resize_concat':
            inputs = [inputs[i] for i in self.in_index]
            upsampled_inputs = [
                resize(
                    input=x,
                    size=inputs[0].shape[2:],
                    mode='bilinear',
                    align_corners=self.align_corners) for x in inputs
            ]
            inputs = torch.cat(upsampled_inputs, dim=1)
        elif self.input_transform == 'multiple_select':
            inputs = [inputs[i] for i in self.in_index]
        else:
            inputs = inputs[self.in_index]

        return inputs

    def forward(self, inputs1, inputs2):
        #Transforming encoder features (select layers)
        x_1 = self._transform_inputs(inputs1)  # len=4, 1/2, 1/4, 1/8, 1/16
        x_2 = self._transform_inputs(inputs2)  # len=4, 1/2, 1/4, 1/8, 1/16

        #img1 and img2 features
        c1_1, c2_1, c3_1, c4_1 = x_1
        c1_2, c2_2, c3_2, c4_2 = x_2

        ############## MLP decoder on C1-C4 ###########
        n, _, h, w = c4_1.shape

        outputs = []
        # Stage 4: x1/32 scale
        _c4_1 = self.linear_c4(c4_1).permute(0,2,1).reshape(n, -1, c4_1.shape[2], c4_1.shape[3])
        _c4_2 = self.linear_c4(c4_2).permute(0,2,1).reshape(n, -1, c4_2.shape[2], c4_2.shape[3])
        _c4   = self.diff_c4(torch.cat((_c4_1, _c4_2), dim=1))
        p_c4  = self.make_pred_c4(_c4)
        outputs.append(p_c4)
        _c4_up= resize(_c4, size=c1_2.size()[2:], mode='bilinear', align_corners=False)

        # Stage 3: x1/16 scale
        _c3_1 = self.linear_c3(c3_1).permute(0,2,1).reshape(n, -1, c3_1.shape[2], c3_1.shape[3])
        _c3_2 = self.linear_c3(c3_2).permute(0,2,1).reshape(n, -1, c3_2.shape[2], c3_2.shape[3])
        _c3   = self.diff_c3(torch.cat((_c3_1, _c3_2), dim=1)
                            ) + F.interpolate(
                                _c4, scale_factor=2, mode="bilinear")
        p_c3  = self.make_pred_c3(_c3)
        outputs.append(p_c3)
        _c3_up= resize(_c3, size=c1_2.size()[2:], mode='bilinear', align_corners=False)

        # Stage 2: x1/8 scale
        _c2_1 = self.linear_c2(c2_1).permute(0,2,1).reshape(n, -1, c2_1.shape[2], c2_1.shape[3])
        _c2_2 = self.linear_c2(c2_2).permute(0,2,1).reshape(n, -1, c2_2.shape[2], c2_2.shape[3])
        _c2   = self.diff_c2(torch.cat((_c2_1, _c2_2), dim=1)) + F.interpolate(_c3, scale_factor=2, mode="bilinear")
        p_c2  = self.make_pred_c2(_c2)
        outputs.append(p_c2)
        _c2_up= resize(_c2, size=c1_2.size()[2:], mode='bilinear', align_corners=False)

        # Stage 1: x1/4 scale
        _c1_1 = self.linear_c1(c1_1).permute(0,2,1).reshape(n, -1, c1_1.shape[2], c1_1.shape[3])
        _c1_2 = self.linear_c1(c1_2).permute(0,2,1).reshape(n, -1, c1_2.shape[2], c1_2.shape[3])
        _c1   = self.diff_c1(torch.cat((_c1_1, _c1_2), dim=1)) + F.interpolate(_c2, scale_factor=2, mode="bilinear")
        p_c1  = self.make_pred_c1(_c1)
        outputs.append(p_c1)

        #Linear Fusion of difference image from all scales
        _c = self.linear_fuse(
            torch.cat(
                (_c4_up, _c3_up, _c2_up, _c1),
                dim=1
                )
            )

        # #Dropout
        # if dropout_ratio > 0:
        #     self.dropout = nn.Dropout2d(dropout_ratio)
        # else:
        #     self.dropout = None

        #Upsampling x2 (x1/2 scale)
        x = self.convd2x(_c)
        #Residual block
        x = self.dense_2x(x)
        #Upsampling x2 (x1 scale)
        x = self.convd1x(x)
        #Residual block
        x = self.dense_1x(x)

        #Final prediction
        cp = self.change_probability(x)
        
        outputs.append(cp)

        if self.output_softmax:
            temp = outputs
            outputs = []
            for pred in temp:
                outputs.append(self.active(pred))

        return outputs
    





# class DecoderTransformer_v3_Mod(nn.Module):
class DualDecoder(nn.Module):
    """
    Transformer Decoder
    """
    def __init__(self, input_transform='multiple_select', in_index=[0, 1, 2, 3], align_corners=True, 
                    in_channels = [32, 64, 128, 256], embedding_dim= 64, output_nc=2, 
                    decoder_softmax = False, feature_strides=[2, 4, 8, 16],fusion=""):#Not using fusion here
        super(DualDecoder, self).__init__()
        #assert
        assert len(feature_strides) == len(in_channels)
        assert min(feature_strides) == feature_strides[0]
        
        #settings
        self.feature_strides = feature_strides
        self.input_transform = input_transform
        self.in_index        = in_index
        self.align_corners   = align_corners
        self.in_channels     = in_channels
        self.embedding_dim   = embedding_dim
        self.output_nc       = output_nc
        c1_in_channels, c2_in_channels, c3_in_channels, c4_in_channels = self.in_channels

        #MLP decoder heads
        self.linear_c4 = MLP(input_dim=c4_in_channels, embed_dim=self.embedding_dim)
        self.linear_c3 = MLP(input_dim=c3_in_channels, embed_dim=self.embedding_dim)
        self.linear_c2 = MLP(input_dim=c2_in_channels, embed_dim=self.embedding_dim)
        self.linear_c1 = MLP(input_dim=c1_in_channels, embed_dim=self.embedding_dim)

        #convolutional Difference Modules
        self.diff_c4   = conv_diff(in_channels=2*self.embedding_dim, out_channels=self.embedding_dim)
        self.diff_c3   = conv_diff(in_channels=2*self.embedding_dim, out_channels=self.embedding_dim)
        self.diff_c2   = conv_diff(in_channels=2*self.embedding_dim, out_channels=self.embedding_dim)
        self.diff_c1   = conv_diff(in_channels=2*self.embedding_dim, out_channels=self.embedding_dim)

        self.diff_c4_mod = conv_diff(in_channels=self.embedding_dim, out_channels=self.embedding_dim)
        self.diff_c3_mod = conv_diff(in_channels=self.embedding_dim, out_channels=self.embedding_dim)
        self.diff_c2_mod = conv_diff(in_channels=self.embedding_dim, out_channels=self.embedding_dim)
        self.diff_c1_mod = conv_diff(in_channels=self.embedding_dim, out_channels=self.embedding_dim)

        #taking outputs from middle of the encoder
        self.make_pred_c4 = make_prediction(in_channels=self.embedding_dim, out_channels=self.output_nc)
        self.make_pred_c3 = make_prediction(in_channels=self.embedding_dim, out_channels=self.output_nc)
        self.make_pred_c2 = make_prediction(in_channels=self.embedding_dim, out_channels=self.output_nc)
        self.make_pred_c1 = make_prediction(in_channels=self.embedding_dim, out_channels=self.output_nc)

        #Final linear fusion layer
        self.linear_fuse = nn.Sequential(
            nn.Conv2d(
                in_channels=self.embedding_dim*len(in_channels), 
                out_channels=self.embedding_dim,
                kernel_size=1
                ),
            nn.BatchNorm2d(
                self.embedding_dim
                )
        )

        #Final predction head
        self.convd2x    = UpsampleConvLayer(self.embedding_dim, self.embedding_dim, kernel_size=4, stride=2)
        self.dense_2x   = nn.Sequential( ResidualBlock(self.embedding_dim))
        self.convd1x    = UpsampleConvLayer(self.embedding_dim, self.embedding_dim, kernel_size=4, stride=2)
        self.dense_1x   = nn.Sequential( ResidualBlock(self.embedding_dim))
        self.change_probability = ConvLayer(self.embedding_dim, self.output_nc, kernel_size=3, stride=1, padding=1)
        
        #Final activation
        self.output_softmax     = decoder_softmax
        self.active             = nn.Sigmoid() 

    def _transform_inputs(self, inputs):
        """Transform inputs for decoder.
        Args:
            inputs (list[Tensor]): List of multi-level img features.
        Returns:
            Tensor: The transformed inputs
        """

        if self.input_transform == 'resize_concat':
            inputs = [inputs[i] for i in self.in_index]
            upsampled_inputs = [
                resize(
                    input=x,
                    size=inputs[0].shape[2:],
                    mode='bilinear',
                    align_corners=self.align_corners) for x in inputs
            ]
            inputs = torch.cat(upsampled_inputs, dim=1)
        elif self.input_transform == 'multiple_select':
            inputs = [inputs[i] for i in self.in_index]
        else:
            inputs = inputs[self.in_index]

        return inputs

    def forward(self, inputs):
        x_1 = self._transform_inputs(inputs)  

        outputs = []

        c1_1, c2_1, c3_1, c4_1 = x_1

        ############## MLP decoder on C1-C4 ###########
        n, _, h, w = c4_1.shape

        _c4_1 = self.linear_c4(c4_1).permute(0,2,1).reshape(n, -1, c4_1.shape[2], c4_1.shape[3])
        _c4 = self.diff_c4_mod(_c4_1)
        p_c4  = self.make_pred_c4(_c4)
        outputs.append(p_c4)

        _c3_1 = self.linear_c3(c3_1).permute(0,2,1).reshape(n, -1, c3_1.shape[2], c3_1.shape[3])
        _c3   = self.diff_c3_mod(_c3_1) + F.interpolate(_c4, scale_factor=2, mode="bilinear")
        p_c3  = self.make_pred_c3(_c3)
        outputs.append(p_c3)

        _c2_1 = self.linear_c2(c2_1).permute(0,2,1).reshape(n, -1, c2_1.shape[2], c2_1.shape[3])
        _c2   = self.diff_c2_mod(_c2_1) + F.interpolate(_c3, scale_factor=2, mode="bilinear")
        p_c2  = self.make_pred_c2(_c2)
        outputs.append(p_c2)

        _c1_1 = self.linear_c1(c1_1).permute(0,2,1).reshape(n, -1, c1_1.shape[2], c1_1.shape[3])
        _c1   = self.diff_c1_mod(_c1_1) + F.interpolate(_c2, scale_factor=2, mode="bilinear")  
        p_c1  = self.make_pred_c1(_c1)  
        outputs.append(p_c1)    

        _c4_up= resize(_c4, size=c1_1.size()[2:], mode='bilinear', align_corners=False)        
        _c3_up= resize(_c3, size=c1_1.size()[2:], mode='bilinear', align_corners=False)        
        _c2_up= resize(_c2, size=c1_1.size()[2:], mode='bilinear', align_corners=False)
        
        _c = torch.cat((_c4_up, _c3_up, _c2_up, _c1), dim=1)
        _c = self.linear_fuse(_c)

        x = self.convd2x(_c) 
        x = self.dense_2x(x)
        x = self.convd1x(x)
        x = self.dense_1x(x) 

        #Final prediction
        cp = self.change_probability(x)  
        outputs.append(cp)

        # if self.output_softmax:
        #     return self.active(cp)
        # else:
        #     return cp
        if self.output_softmax:
            temp = outputs
            outputs = []
            for pred in temp:
                outputs.append(self.active(pred))

        return outputs


def attentionFusion(x1, x2):
    n, c, h, w = x1.size()
    x1_flat = x1.view(n, c, -1)  
    x2_flat = x2.view(n, c, -1)  

    attention_scores = torch.bmm(x1_flat.transpose(1, 2), x2_flat)

    attention_weights = F.softmax(attention_scores, dim=-1)

    x1_weighted = torch.bmm(x1_flat, attention_weights.transpose(1, 2))  
    x2_weighted = torch.bmm(x2_flat, attention_weights) 

    x1_weighted = x1_weighted.view(n, c, h, w)
    x2_weighted = x2_weighted.view(n, c, h, w)

    fused = x1_weighted + x2_weighted  

    return fused


def fuse(fusion, x1, x2):
    if fusion == "concat":
        return torch.cat((x1, x2), dim=1)
    elif fusion == "sum":
        return x1 + x2
    elif fusion == "average":
        return 0.5 * (x1 + x2)
    elif fusion == "subtraction":
        return x1 - x2
    elif fusion == "abs_diff":
        return torch.abs(x1 - x2)
    elif fusion == "attention":
        return attentionFusion(x1,x2)
    else:
        raise Exception(f"Fusion method provided ('{fusion}') is not allowed")

class Decoder(nn.Module):
    def __init__(self, 
                input_transform='multiple_select', 
                in_index=[0, 1, 2, 3], 
                align_corners=True, 
                in_channels = [32, 64, 128, 256], 
                embedding_dim= 64, 
                output_nc=2, 
                decoder_softmax = False, 
                feature_strides=[2, 4, 8, 16],
                fusion="concat"
                ):
        super(Decoder, self).__init__()
        assert len(feature_strides) == len(in_channels)
        assert min(feature_strides) == feature_strides[0]
        
        self.feature_strides = feature_strides
        self.input_transform = input_transform
        self.in_index        = in_index
        self.align_corners   = align_corners
        self.in_channels     = in_channels
        self.embedding_dim   = embedding_dim
        self.output_nc       = output_nc
        c1_in_channels, c2_in_channels, c3_in_channels, c4_in_channels = self.in_channels
        self.fusion = fusion

        print("FUSION METHOD: ",self.fusion)

        #MLP decoder heads
        self.linear_c4 = MLP(input_dim=c4_in_channels, embed_dim=self.embedding_dim)
        self.linear_c3 = MLP(input_dim=c3_in_channels, embed_dim=self.embedding_dim)
        self.linear_c2 = MLP(input_dim=c2_in_channels, embed_dim=self.embedding_dim)
        self.linear_c1 = MLP(input_dim=c1_in_channels, embed_dim=self.embedding_dim)

        #convolutional Difference Modules
        inch = 2*self.embedding_dim if fusion=="concat" else self.embedding_dim
        self.diff_c4   = conv_diff(in_channels=inch, out_channels=self.embedding_dim)
        self.diff_c3   = conv_diff(in_channels=inch, out_channels=self.embedding_dim)
        self.diff_c2   = conv_diff(in_channels=inch, out_channels=self.embedding_dim)
        self.diff_c1   = conv_diff(in_channels=inch, out_channels=self.embedding_dim)

        #taking outputs from middle of the encoder
        self.make_pred_c4 = make_prediction(in_channels=self.embedding_dim, out_channels=self.output_nc)
        self.make_pred_c3 = make_prediction(in_channels=self.embedding_dim, out_channels=self.output_nc)
        self.make_pred_c2 = make_prediction(in_channels=self.embedding_dim, out_channels=self.output_nc)
        self.make_pred_c1 = make_prediction(in_channels=self.embedding_dim, out_channels=self.output_nc)

        #Final linear fusion layer
        self.linear_fuse = nn.Sequential(
            nn.Conv2d(   in_channels=self.embedding_dim*len(in_channels), out_channels=self.embedding_dim,
                                        kernel_size=1),
            nn.BatchNorm2d(self.embedding_dim)
        )

        #Final predction head
        self.convd2x    = UpsampleConvLayer(self.embedding_dim, self.embedding_dim, kernel_size=4, stride=2)
        self.dense_2x   = nn.Sequential( ResidualBlock(self.embedding_dim))
        self.convd1x    = UpsampleConvLayer(self.embedding_dim, self.embedding_dim, kernel_size=4, stride=2)
        self.dense_1x   = nn.Sequential( ResidualBlock(self.embedding_dim))
        self.change_probability = ConvLayer(self.embedding_dim, self.output_nc, kernel_size=3, stride=1, padding=1)
        
        #Final activation
        self.output_softmax     = decoder_softmax
        self.active             = nn.Sigmoid() 

    def _transform_inputs(self, inputs):
        """Transform inputs for decoder.
        Args:
            inputs (list[Tensor]): List of multi-level img features.
        Returns:
            Tensor: The transformed inputs
        """

        if self.input_transform == 'resize_concat':
            inputs = [inputs[i] for i in self.in_index]
            upsampled_inputs = [
                resize(
                    input=x,
                    size=inputs[0].shape[2:],
                    mode='bilinear',
                    align_corners=self.align_corners) for x in inputs
            ]
            inputs = torch.cat(upsampled_inputs, dim=1)
        elif self.input_transform == 'multiple_select':
            inputs = [inputs[i] for i in self.in_index]
        else:
            inputs = inputs[self.in_index]

        return inputs

    def forward(self, inputs1, inputs2):
        #Transforming encoder features (select layers)
        x_1 = self._transform_inputs(inputs1)  # len=4, 1/2, 1/4, 1/8, 1/16
        x_2 = self._transform_inputs(inputs2)  # len=4, 1/2, 1/4, 1/8, 1/16

        #img1 and img2 features
        c1_1, c2_1, c3_1, c4_1 = x_1
        c1_2, c2_2, c3_2, c4_2 = x_2

        ############## MLP decoder on C1-C4 ###########
        n, _, h, w = c4_1.shape

        outputs = []
        # Stage 4: x1/32 scale
        _c4_1 = self.linear_c4(c4_1).permute(0,2,1).reshape(n, -1, c4_1.shape[2], c4_1.shape[3])
        _c4_2 = self.linear_c4(c4_2).permute(0,2,1).reshape(n, -1, c4_2.shape[2], c4_2.shape[3])
        # _c4   = self.diff_c4(self.fuse(_c4_1, _c4_2))#torch.cat((_c4_1, _c4_2), dim=1)
        _c4   = self.diff_c4(fuse(self.fusion,_c4_1, _c4_2))#torch.cat((_c4_1, _c4_2), dim=1)
        p_c4  = self.make_pred_c4(_c4)
        outputs.append(p_c4)
        _c4_up= resize(_c4, size=c1_2.size()[2:], mode='bilinear', align_corners=False)

        # Stage 3: x1/16 scale
        _c3_1 = self.linear_c3(c3_1).permute(0,2,1).reshape(n, -1, c3_1.shape[2], c3_1.shape[3])
        _c3_2 = self.linear_c3(c3_2).permute(0,2,1).reshape(n, -1, c3_2.shape[2], c3_2.shape[3])
        _c3   = self.diff_c3(fuse(self.fusion,_c3_1, _c3_2)) + F.interpolate(_c4, scale_factor=2, mode="bilinear")
            
        p_c3  = self.make_pred_c3(_c3)
        outputs.append(p_c3)
        _c3_up= resize(_c3, size=c1_2.size()[2:], mode='bilinear', align_corners=False)

        # Stage 2: x1/8 scale
        _c2_1 = self.linear_c2(c2_1).permute(0,2,1).reshape(n, -1, c2_1.shape[2], c2_1.shape[3])
        _c2_2 = self.linear_c2(c2_2).permute(0,2,1).reshape(n, -1, c2_2.shape[2], c2_2.shape[3])
        _c2   = self.diff_c2(
            # torch.cat((_c2_1, _c2_2), dim=1)
            # self.fuse(_c2_1, _c2_2)
            fuse(self.fusion,_c2_1, _c2_2)
            ) + F.interpolate(_c3, scale_factor=2, mode="bilinear")
        p_c2  = self.make_pred_c2(_c2)
        outputs.append(p_c2)
        _c2_up= resize(_c2, size=c1_2.size()[2:], mode='bilinear', align_corners=False)

        # Stage 1: x1/4 scale
        _c1_1 = self.linear_c1(c1_1).permute(0,2,1).reshape(n, -1, c1_1.shape[2], c1_1.shape[3])
        _c1_2 = self.linear_c1(c1_2).permute(0,2,1).reshape(n, -1, c1_2.shape[2], c1_2.shape[3])
        _c1   = self.diff_c1(
            # torch.cat((_c1_1, _c1_2), dim=1)
            # self.fuse(_c1_1, _c1_2)
            fuse(self.fusion,_c1_1, _c1_2)
            ) + F.interpolate(_c2, scale_factor=2, mode="bilinear")
        p_c1  = self.make_pred_c1(_c1)
        outputs.append(p_c1)

        #Linear Fusion of difference image from all scales
        _c = self.linear_fuse(
            torch.cat(
                (_c4_up, _c3_up, _c2_up, _c1),
                dim=1
                )
            )

        # #Dropout
        # if dropout_ratio > 0:
        #     self.dropout = nn.Dropout2d(dropout_ratio)
        # else:
        #     self.dropout = None

        #Upsampling x2 (x1/2 scale)
        x = self.convd2x(_c)
        #Residual block
        x = self.dense_2x(x)
        #Upsampling x2 (x1 scale)
        x = self.convd1x(x)
        #Residual block
        x = self.dense_1x(x)

        #Final prediction
        cp = self.change_probability(x)
        
        outputs.append(cp)

        if self.output_softmax:
            temp = outputs
            outputs = []
            for pred in temp:
                outputs.append(self.active(pred))

        return outputs