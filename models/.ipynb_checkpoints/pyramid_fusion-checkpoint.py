#!/usr/bin/env python
# coding: utf-8
# %%
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models
import collections
import math
import sys
sys.path.append("/mnt/batch/tasks/shared/LS_root/mounts/clusters/objloc/code/pyramid-fuse")
# sys.setrecursionlimit(10000000)
import Utils
from Utils.CubePad import CustomPad
import numpy as np

# %%
pre_trained = torch.load("BiFuse_Pretrained.pkl")


# %%
class Unpool(nn.Module):
    # Unpool: 2*2 unpooling with zero padding
    def __init__(self, num_channels, stride=2):
        super(Unpool, self).__init__()

        self.num_channels = num_channels
        self.stride = stride

        # create kernel [1, 0; 0, 0]
        # currently not compatible with running on CPU
        self.weights = torch.autograd.Variable(
            torch.zeros(num_channels, 1, stride, stride))
        self.weights[:, :, 0, 0] = 1

    def forward(self, x):
        return F.conv_transpose2d(x, self.weights.cuda(), stride=self.stride, groups=self.num_channels)


# %%
class Decoder(nn.Module):
    # Decoder is the base class for all decoders

    names = ['pyramid1', 'pyramid2', 'pyramid3', 'pyramid4']

    def __init__(self):
        super(Decoder, self).__init__()

        self.layer1 = None
        self.layer2 = None
        self.layer3 = None
        self.layer4 = None

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x


# %%


class PSPModule(nn.Module):
    def __init__(self, features, out_features=1024, sizes=(1, 2, 3, 6)):
        super().__init__()
        self.stages = []
        self.stages = nn.ModuleList([self._make_stage(features, size) for size in sizes])
        self.bottleneck = nn.Conv2d(features * (len(sizes) + 1), out_features, kernel_size=1)
        self.relu = nn.ReLU()

    def _make_stage(self, features, size):
        prior = nn.AdaptiveAvgPool2d(output_size=(size, size))
        conv = nn.Conv2d(features, features, kernel_size=1, bias=False)
        return nn.Sequential(prior, conv)

    def forward(self, feats):
        h, w = feats.size(2), feats.size(3)
        priors = [F.upsample(input=stage(feats), size=(h, w), mode='bilinear') for stage in self.stages] + [feats]
        bottle = self.bottleneck(torch.cat(priors, 1))
        return self.relu(bottle)


# %%


class PSPUpsample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.PReLU()
        )

    def forward(self, x):
        h, w = 2 * x.size(2), 2 * x.size(3)
        p = F.upsample(input=x, size=(h, w), mode='bilinear')
        return self.conv(p)


# %%


class PSPNet(nn.Module):
    def __init__(self, n_classes=21, sizes=(1, 2, 3, 6), psp_size=1024, deep_features_size=1024):
        super().__init__()
        self.psp = PSPModule(psp_size, 1024, sizes)
        self.drop_1 = nn.Dropout2d(p=0.3)

        self.up_1 = PSPUpsample(1024, 256)
        self.up_2 = PSPUpsample(256, 64)
        self.up_3 = PSPUpsample(64, 64)

        self.drop_2 = nn.Dropout2d(p=0.15)
        self.final = nn.Sequential(
            nn.Conv2d(64, n_classes, kernel_size=1),
            nn.LogSoftmax()
        )

        self.classifier = nn.Sequential(
            nn.Linear(deep_features_size, 256),
            nn.ReLU(),
            nn.Linear(256, n_classes)
        )

    def forward(self, x):
#         print(x.shape)
        f = x
        class_f = x
        p = self.psp(f)
        p = self.drop_1(p)

        p = self.up_1(p)
        p = self.drop_2(p)

        p = self.up_2(p)
        p = self.drop_2(p)

        p = self.up_3(p)
        p = self.drop_2(p)

        auxiliary = F.adaptive_max_pool2d(input=class_f, output_size=(1, 1)).view(-1, class_f.size(1))
   
        return self.final(p), self.classifier(auxiliary)


# %%
#Using this
class UpProj(Decoder):
    # UpProj decoder consists of 4 upproj modules with decreasing number of channels and increasing feature map size

    class UpProjModule(nn.Module):
        # UpProj module has two branches, with a Unpool at the start and a ReLu at the end
        #   upper branch: 5*5 conv -> batchnorm -> ReLU -> 3*3 conv -> batchnorm
        #   bottom branch: 5*5 conv -> batchnorm

        def __init__(self, in_channels, out_channels=None, padding=None):
            super(UpProj.UpProjModule, self).__init__()
            if out_channels is None:
                out_channels = in_channels//2
            self.pad_3 = padding(1)
            self.pad_5 = padding(2)

            self.unpool = Unpool(in_channels)
            self.upper_branch = nn.Sequential(collections.OrderedDict([
                ('pad1', CustomPad(self.pad_5)),
                ('conv1',      nn.Conv2d(in_channels, out_channels,
                                         kernel_size=5, stride=1, padding=0, bias=False)),
                ('batchnorm1', nn.BatchNorm2d(out_channels)),
                ('relu',      nn.ReLU()),
                ('pad2', CustomPad(self.pad_3)),
                ('conv2',      nn.Conv2d(out_channels, out_channels,
                                         kernel_size=3, stride=1, padding=0, bias=False)),
                ('batchnorm2', nn.BatchNorm2d(out_channels)),
            ]))
            self.bottom_branch = nn.Sequential(collections.OrderedDict([
                ('pad', CustomPad(self.pad_5)),
                ('conv',      nn.Conv2d(in_channels, out_channels,
                                        kernel_size=5, stride=1, padding=0, bias=False)),
                ('batchnorm', nn.BatchNorm2d(out_channels)),
            ]))
            self.relu = nn.ReLU()
            s
        def forward(self, x):
            x = self.unpool(x)
            x1 = self.upper_branch(x)
            x2 = self.bottom_branch(x)
            x = x1 + x2
            x = self.relu(x)
            return x

    def __init__(self, in_channels, padding):
        super(UpProj, self).__init__()
        self.padding = getattr(Utils.CubePad, padding)
        self.layer1 = self.UpProjModule(in_channels   , padding=self.padding)
        self.layer2 = self.UpProjModule(in_channels//2, padding=self.padding)
        self.layer3 = self.UpProjModule(in_channels//4, padding=self.padding)
        self.layer4 = self.UpProjModule(in_channels//8, padding=self.padding)


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
                        nn.Conv2d(42, 32, kernel_size=3, stride=1, padding=1, bias=False),
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
                        nn.Conv2d(96, 32, kernel_size=3, stride=1, padding=1, bias=False),
                        nn.BatchNorm2d(32),
                        nn.ReLU(inplace=True),
                        nn.Conv2d(32, 21, kernel_size=3, stride=1, padding=1, bias=False)
                        )
        self.bilinear_1 = nn.UpsamplingBilinear2d(size=(64,128))
        self.bilinear_2 = nn.UpsamplingBilinear2d(size=(128,256))
    def forward(self, inputs):
        x = inputs
#         print("\n Entering Refine Model \n")
#         print("X_Shape: ", x.shape)
        
        out_1 = self.refine_1(x)
#         print("out_1_Shape: ", out_1.shape)
        
        out_2 = self.refine_2(out_1)
#         print("Out_2_Shape: ", out_2.shape)
        
        deconv_out1 = self.deconv_1(out_2)
#         print("Deconv_out1_Shape: ", deconv_out1.shape)
        
        up_1 = self.bilinear_1(out_2)
#         print("up1_Shape: ", up_1.shape)
        
        deconv_out2 = self.deconv_2(torch.cat((deconv_out1, up_1), dim = 1))
#         print("Deconv_out2_Shape: ", deconv_out2.shape)
        
        up_2 = self.bilinear_2(out_1)
#         print("Up_2_Shape: ", up_2.shape)
        
        out_3 = self.refine_3(torch.cat((deconv_out2, up_2), dim = 1))
#         print("out_3_Shape: ", out_3.shape)

        return out_3  


# %%
class PyFuse(nn.Module):
    def __init__(self, layers, output_size=None, in_channels=3, pretrained=True):
        super(PyFuse, self).__init__()
        bs = 1
#Initializing the models
        self.equi_model = fusion_ResNet(
            bs, layers, (512, 1024), 3, pretrained, padding='ZeroPad')
        print(self.equi_model)
        self.cube_model = fusion_ResNet(
            bs*6, layers, (256, 256), 3, pretrained, padding='SpherePad')
        self.equi_psp = PSPNet()
        self.cube_psp = PSPNet()
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
    def forward(self, equi, fusion=False):
#going from E2C
        cube = self.ce.E2C(equi)
#Applying the pre-processing block to deal with distortion
        feat_equi = self.equi_model.pre_encoder2(equi)
#Check this one out
        feat_cube = self.cube_model.pre_encoder(cube)
#         print("Equi_shape: ", equi.shape)
        print("Pyramid encoder input-Equi:", feat_equi.shape)
        print("Pyramid encoder input-Cube:", feat_cube.shape)
        

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
                
#         print("Pyramid encoder exit-Equi:", feat_equi.shape)
#         print("Pyramid encoder exit-Cube:", feat_cube.shape)
        
        feat_equi = self.equi_psp(feat_equi)
        feat_equi = feat_equi[0]
        feat_cube = self.cube_psp(feat_cube)
        feat_cube = feat_cube[0]
         
#         print("Pyramid decoder exit-Equi:", feat_equi.shape)
#         print("Pyramid decoder exit-Cube:", feat_cube.shape)
        
        feat_cube = self.ce.C2E(feat_cube)
        feat_cat = torch.cat((feat_equi, feat_cube), dim = 1)
        
#         print("\n C2E and Cat shapes \n")
#         print("C2E-Cube: ", feat_cube.shape)
#         print("Cat: ", feat_cat.shape)
        
        refine_final = self.refine_model(feat_cat)
        
        return refine_final

