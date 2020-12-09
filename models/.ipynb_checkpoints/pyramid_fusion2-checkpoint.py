#!/usr/bin/env python
# coding: utf-8
# %%

# %%
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models
import collections
import math
import sys
sys.path.append("/mnt/batch/tasks/shared/LS_root/mounts/clusters/pyfuse/code/pyramid-fuse")
# sys.setrecursionlimit(10000000)
import Utils
from Utils.CubePad import CustomPad
from utils_seg.helpers import initialize_weights
import numpy as np


# %%
def weights_init(m):
#     print(m)
    # Initialize filters with Gaussian random weights
    if isinstance(m, nn.Conv2d):
        n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        m.weight.data.normal_(0, math.sqrt(2. / n))
        if m.bias is not None:
            m.bias.data.zero_()
    elif isinstance(m, nn.ConvTranspose2d):
        n = m.kernel_size[0] * m.kernel_size[1] * m.in_channels
        m.weight.data.normal_(0, math.sqrt(2. / n))
        if m.bias is not None:
            m.bias.data.zero_()
    elif isinstance(m, nn.BatchNorm2d):
        m.weight.data.fill_(1)
        m.bias.data.zero_()


# %%
class _PSPModule(nn.Module):
    def __init__(self, in_channels, bin_sizes, norm_layer):
        super(_PSPModule, self).__init__()
        out_channels = in_channels // len(bin_sizes)
        self.stages = nn.ModuleList([self._make_stages(in_channels, out_channels, b_s, norm_layer) 
                                                        for b_s in bin_sizes])
        self.bottleneck = nn.Sequential(
            nn.Conv2d(in_channels+(out_channels * len(bin_sizes)), out_channels, 
                                    kernel_size=3, padding=1, bias=False),
            norm_layer(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1)
        )

    def _make_stages(self, in_channels, out_channels, bin_sz, norm_layer):
        prior = nn.AdaptiveAvgPool2d(output_size=bin_sz)
        conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        bn = norm_layer(out_channels)
        relu = nn.ReLU(inplace=True)
        return nn.Sequential(prior, conv, bn, relu)
    
    def forward(self, features):
        h, w = features.size()[2], features.size()[3]
        pyramids = [features]
        pyramids.extend([F.interpolate(stage(features), size=(h, w), mode='bilinear', 
                                        align_corners=True) for stage in self.stages])
        output = self.bottleneck(torch.cat(pyramids, dim=1))
        return output


# %%
class PSPNet(nn.Module):
    def __init__(self, num_classes=21, use_aux=True):
        super(PSPNet, self).__init__()
        # TODO: Use synch batchnorm
        norm_layer = nn.InstanceNorm2d
#         model = getattr(resnet, backbone)(pretrained, norm_layer=norm_layer, )
#         m_out_sz = model.fc.in_features
        self.use_aux = use_aux 

#         self.initial = nn.Sequential(*list(model.children())[:4])
#         if in_channels != 3:
#             self.initial[0] = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
#         self.initial = nn.Sequential(*self.initial)
        
#         self.layer1 = model.layer1
#         self.layer2 = model.layer2
#         self.layer3 = model.layer3
#         self.layer4 = model.layer4

        self.master_branch = nn.Sequential(
            _PSPModule(1024, bin_sizes=[1, 2, 3, 6], norm_layer=norm_layer),
            nn.Conv2d(1024//4, num_classes, kernel_size=1)
        )

        self.auxiliary_branch = nn.Sequential(
            nn.Conv2d(1024, 1024//2, kernel_size=3, padding=1, bias=False),
            norm_layer(1024//2),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1),
            nn.Conv2d(1024//2, num_classes, kernel_size=1)
        )
        initialize_weights(self.master_branch, self.auxiliary_branch)
#         if freeze_bn: self.freeze_bn()
#         if freeze_backbone: 
#             set_trainable([self.initial, self.layer1, self.layer2, self.layer3, self.layer4], False)

#Equi - [bs, [128, 256]
#cubemap [64, 64]
    def forward(self, x):
#         input_size = (128, 256)
#         if x.shape[0]%6==0:
#             input_size = (8*x.size()[2], 8*x.size()[3])
        input_size = (512, 1024)
#         print("PSP Input: ", x.shape)
        if x.shape[0]%6==0:
            input_size = (32*x.size()[2], 32*x.size()[3])
#         x = self.initial(x)
#         x = self.layer1(x)
#         x = self.layer2(x)
#         x_aux = self.layer3(x)
#         x = self.layer4(x_aux)
        x_aux = x
        output = self.master_branch(x)
        output = F.interpolate(output, size=input_size, mode='bilinear')
        output = output[:, :, :input_size[0], :input_size[1]]

        if self.training and self.use_aux:
            aux = self.auxiliary_branch(x_aux)
            aux = F.interpolate(aux, size=input_size, mode='bilinear')
            aux = aux[:, :, :input_size[0], :input_size[1]]
            return output
        print("PSP Output:", output.shape)
        return output


# %%
def e2c(equirectangular):
    cube = Utils.Equirec2Cube.ToCubeTensor(equirectangular.cuda())
    return cube

def c2e(cube):
    equirectangular = Utils.Cube2Equirec.ToEquirecTensor(cube.cuda())
    return equirectangular


# %%
class PreprocBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size_lst, stride=2):
        super(PreprocBlock, self).__init__()
        assert len(kernel_size_lst) == 4 and out_channels % 4 == 0
        self.lst = nn.ModuleList([])

        for (h, w) in kernel_size_lst:
            padding = (h//2, w//2)
            tmp = nn.Sequential(
                        nn.Conv2d(in_channels, out_channels//4, kernel_size=(h,w), stride=stride, padding=padding),
                        nn.BatchNorm2d(out_channels//4),
                        nn.ReLU(inplace=True)
                    )
            self.lst.append(tmp)

    def forward(self, x):
        out = []
        for conv in self.lst:
            out.append(conv(x))
        out = torch.cat(out, dim=1)
        return out


# %%
class fusion_ResNet(nn.Module):
    _output_size_init = (256, 256)
    def __init__(self, bs, layers, output_size=(256, 256), in_channels=3, pretrained=True, padding='ZeroPad'):

        if layers not in [18, 34, 50, 101, 152]:
            raise RuntimeError(
                'Only 18, 34, 50, 101, and 152 layer model are defined for ResNet. Got {}'.format(layers))

        super(fusion_ResNet, self).__init__()
        self.padding = getattr(Utils.CubePad, padding)
        self.pad_7 = self.padding(3)
        self.pad_3 = self.padding(1)
        try: from . import resnet
        except: import resnet
        pretrained_model = getattr(resnet, 'resnet%d'%layers)(pretrained=pretrained, padding=padding)

        if in_channels == 3:
            self.conv1 = pretrained_model._modules['conv1']
            self.bn1 = pretrained_model._modules['bn1']
        else:
            self.conv1 = nn.Conv2d(
                in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
            self.bn1 = nn.BatchNorm2d(64)
            weights_init(self.conv1)
            weights_init(self.bn1)

        self.output_size = output_size
        if output_size == None:
            output_size = _output_size_init
        else:
            assert isinstance(output_size, tuple)
        self.relu = pretrained_model._modules['relu']
        self.maxpool = pretrained_model._modules['maxpool']
        self.layer1 = pretrained_model._modules['layer1']
        self.layer2 = pretrained_model._modules['layer2']
        self.layer3 = pretrained_model._modules['layer3']
        self.layer4 = pretrained_model._modules['layer4']

        # clear memory
        del pretrained_model

        # define number of intermediate channels
        if layers <= 34:
            num_channels = 512
        elif layers >= 50:
            num_channels = 2048

        self.conv2 = nn.Conv2d(num_channels, num_channels //
                               2, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(num_channels//2)

        self.conv2.apply(weights_init)
        self.bn2.apply(weights_init)
        
        self.pre1 = PreprocBlock(3, 64, [[3, 9], [5, 11], [5, 7], [7, 7]])
        self.pre1.apply(weights_init)

    def forward(self, inputs):
        # resnet
        x = inputs
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x0 = x
        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)

        x = self.conv2(x4)
        x = self.bn2(x)

        return x

    def pre_encoder(self, x):
        x = self.conv1(self.pad_7(x))
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(self.pad_3(x))

        return x
    
    def pre_encoder2(self, x):
        x = self.pre1(x)
        x = self.maxpool(self.pad_3(x))

        return x


# %%
class CETransform(nn.Module):
    def __init__(self):
        super(CETransform, self).__init__()
        equ_h = [512, 128, 64, 32, 16]
        cube_h = [256, 64, 32, 16, 8]

        self.c2e = dict()
        self.e2c = dict()

        for h in equ_h:
            a = Utils.Equirec2Cube(1, h, h*2, h//2, 90)
            self.e2c['(%d,%d)' % (h, h*2)] = a

        for h in cube_h:
            a = Utils.Cube2Equirec(1, h, h*2, h*4)
            self.c2e['(%d)' % (h)] = a

    def E2C(self, x):
#         print(x.shape)
        [bs, c, h, w] = x.shape
        key = '(%d,%d)' % (h, w)
        assert key in self.e2c
        return self.e2c[key].ToCubeTensor(x)

    def C2E(self, x):
#         print(x.shape)
        [bs, c, h, w] = x.shape
        key = '(%d)' % (h)
        assert key in self.c2e and h == w
        return self.c2e[key].ToEquirecTensor(x)

    def forward(self, equi, cube):
        return self.e2c(equi), self.c2e(cube)


# %%
class Refine(nn.Module):
    def __init__(self):
        super(Refine, self).__init__()
        self.refine_1 = nn.Sequential(
                        nn.Conv2d(45, 32, kernel_size=3, stride=1, padding=1, bias=False),
                        nn.BatchNorm2d(32),
                        nn.ReLU(inplace=True),
                        nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1, bias=False),
                        nn.BatchNorm2d(64),
                        nn.ReLU(inplace=True),
                        nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False),
                        nn.BatchNorm2d(64),
                        nn.ReLU(inplace=True)
                        )
        self.refine_2 = nn.Sequential(
                        nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=False),
                        nn.BatchNorm2d(128),
                        nn.ReLU(inplace=True),
                        nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=False),
                        nn.BatchNorm2d(128),
                        nn.ReLU(inplace=True),
                        )
        self.deconv_1 = nn.Sequential(
                        nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1, output_padding=0, groups=1, bias=True, dilation=1),
                        nn.BatchNorm2d(64),
                        nn.LeakyReLU(inplace=True),
                        )
        self.deconv_2 = nn.Sequential(
                        nn.ConvTranspose2d(192, 32, kernel_size=4, stride=2, padding=1, output_padding=0, groups=1, bias=True, dilation=1),
                        nn.BatchNorm2d(32),
                        nn.LeakyReLU(inplace=True),
                        )
        self.refine_3 = nn.Sequential(
                        nn.Conv2d(96, 16, kernel_size=3, stride=1, padding=1, bias=False),
                        nn.BatchNorm2d(16),
                        nn.ReLU(inplace=True),
                        nn.Conv2d(16, 21, kernel_size=3, stride=1, padding=1, bias=False)
                        )
        self.bilinear_1 = nn.UpsamplingBilinear2d(size=(256,512))
        self.bilinear_2 = nn.UpsamplingBilinear2d(size=(512,1024))
    def forward(self, inputs):
        x = inputs
        out_1 = self.refine_1(x)
        out_2 = self.refine_2(out_1)
        deconv_out1 = self.deconv_1(out_2)
        up_1 = self.bilinear_1(out_2)
        deconv_out2 = self.deconv_2(torch.cat((deconv_out1, up_1), dim = 1))
        up_2 = self.bilinear_2(out_1)
        out_3 = self.refine_3(torch.cat((deconv_out2, up_2), dim = 1))

        return out_3                


# %%
class PyFuse(nn.Module):
    def __init__(self, layers, output_size=None, in_channels=3, pretrained=True):
        super(PyFuse, self).__init__()
        bs = 1
#Initializing the models
        self.equi_model = fusion_ResNet(
            bs, layers, (512, 1024), 3, pretrained, padding='ZeroPad')
        self.cube_model = fusion_ResNet(
            bs*6, layers, (256, 256), 3, pretrained, padding='SpherePad')
        self.equi_psp = PSPNet()
        self.cube_psp = PSPNet()
#         self.equi_psp.apply(weights_init)
#         self.cube_psp.apply(weights_init)
#Initializing the refine module
        self.refine_model = Refine()

        if layers <= 34:
            num_channels = 512
        elif layers >= 50:
            num_channels = 2048
            
#         self.equi_decoder = PSPNet(self)
#Change input to equi_conv3
        self.equi_conv3 = nn.Sequential(
                nn.Conv2d(num_channels//32, 1, kernel_size=3, stride=1, padding=1, bias=False),
                nn.UpsamplingBilinear2d(size=(512, 1024))
                )
#Change input to cube_conv3
#         self.cube_decoder = PSPNet(self)
        mypad = getattr(Utils.CubePad, 'SpherePad')
        self.cube_conv3 = nn.Sequential(
                mypad(1),
                nn.Conv2d(num_channels//32, 1, kernel_size=3, stride=1, padding=0, bias=False),
                nn.UpsamplingBilinear2d(size=(256, 256))
                )
#Applying weights so probably using pre-trained models.
#         self.equi_decoder.apply(weights_init)
        self.equi_conv3.apply(weights_init)
#         self.cube_decoder.apply(weights_init)
        self.cube_conv3.apply(weights_init)

#Transformation function going from C2E and E2C
        self.ce = CETransform()
        
        if layers <= 34:
            ch_lst = [64, 64, 128, 256, 512, 256, 128, 64, 32]
        else:
            ch_lst = [64, 256, 512, 1024, 2048, 1024, 512, 256, 128]

#Declaring convolution of e2c, c2e and mask - A module list is used to construct a network. 
        self.conv_e2c = nn.ModuleList([])
        self.conv_c2e = nn.ModuleList([])
        self.conv_mask = nn.ModuleList([])
#A loop to run through ch_list - This is basically convolution.
#Preparing the encoder
        for i in range(5):
            conv_c2e = nn.Sequential(
                        nn.Conv2d(ch_lst[i], ch_lst[i], kernel_size=3, padding=1),
                        nn.ReLU(inplace=True)
                    )
            conv_e2c = nn.Sequential(
                        nn.Conv2d(ch_lst[i], ch_lst[i], kernel_size=3, padding=1),
                        nn.ReLU(inplace=True)
                    )
            conv_mask = nn.Sequential(
                        nn.Conv2d(ch_lst[i]*2, 1, kernel_size=1, padding=0),
                        nn.Sigmoid()
                    )
            self.conv_e2c.append(conv_e2c)
            self.conv_c2e.append(conv_c2e)
            self.conv_mask.append(conv_mask)



        self.grid = Utils.Equirec2Cube(None, 512, 1024, 256, 90).GetGrid()

#Forwarard for FCRN
    def forward(self, equi, fusion=True):
        
#         print("Input to network: ", equi.shape)
#         equi = equi.permute(0, 1, 3, 2)

#going from E2C
        cube = self.ce.E2C(equi)
#         print("Cube E2C(equi): ", cube.shape)
#Applying the pre-processing block to deal with distortion
        feat_equi = self.equi_model.pre_encoder2(equi)

#Check this one out
        feat_cube = self.cube_model.pre_encoder(cube)


#         print("Pyramid encoder input-Equi:", feat_equi.shape)
#         print("Pyramid encoder input-Cube:", feat_cube.shape)
        
#Running the encoder block
        for e in range(5):
            if fusion:
                aaa = self.conv_e2c[e](feat_equi)
                tmp_cube = self.ce.E2C(aaa)
                tmp_equi = self.conv_c2e[e](self.ce.C2E(feat_cube))
                mask_equi = self.conv_mask[e](torch.cat([aaa, tmp_equi], dim=1))
                mask_cube = 1 - mask_equi
                tmp_cube = tmp_cube.clone() * self.ce.E2C(mask_cube)
                tmp_equi = tmp_equi.clone() * mask_equi
            else:
                tmp_cube = 0
                tmp_equi = 0
            feat_cube = feat_cube + tmp_cube
            feat_equi = feat_equi + tmp_equi
            if e < 4:
                feat_cube = getattr(self.cube_model, 'layer%d'%(e+1))(feat_cube)
                feat_equi = getattr(self.equi_model, 'layer%d'%(e+1))(feat_equi)
            else:
                feat_cube = self.cube_model.conv2(feat_cube)
                feat_equi = self.equi_model.conv2(feat_equi)
                feat_cube = self.cube_model.bn2(feat_cube)
                feat_equi = self.equi_model.bn2(feat_equi)
                
        print("Pyramid encoder exit-Equi:", feat_equi.shape)
#         print("Pyramid encoder exit-Cube:", feat_cube.shape)
        
        feat_equi = self.equi_psp(feat_equi)
#         feat_equi = feat_equi[0]
        feat_cube = self.cube_psp(feat_cube)
#         feat_cube = feat_cube[0]
         
#         print("Pyramid decoder exit-Equi:", feat_equi.shape)
#         print("Pyramid decoder exit-Cube:", feat_cube.shape)
        
        feat_cube = self.ce.C2E(feat_cube)
#         feat_cat = torch.cat((equi, feat_equi, feat_cube), dim = 1)
        
#         print("\n C2E and Cat shapes \n")
#         print("C2E-Cube: ", feat_cube.shape)
#         print("Cat: ", feat_cat.shape)
        
#         refine_final = self.refine_model(feat_cat)
        
#         print("Pyfuse output: ", refine_final.shape)
#         return refine_final
        output = [feat_equi.cuda(), feat_cube.cuda()]
        return output

