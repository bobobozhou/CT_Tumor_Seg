import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import torch.utils.model_zoo as model_zoo
from collections import OrderedDict

model_urls = {
    'vgg16': 'https://download.pytorch.org/models/vgg16-397923af.pth'
}

from PIL import Image
import os
import os.path
import numpy as np
import math
import ipdb


""" Scale-invariant Boundary Aware Net (SiBA-Net - DICE Version) """


class SiBANET(nn.Module):
    def __init__(self, num_classes=4):
        super(SiBANET, self).__init__()

        # TODO: load pre-trained weights from VGG16-Net
        # load parts of pre-train
        state_dict_pretrain = model_zoo.load_url(model_urls['vgg16'])

        '''
        Pre-trained VGG weights - 1st block
        '''        
        self.w1_1_ba = Parameter(state_dict_pretrain['features.0.weight'], requires_grad=True)  # weight64_1
        self.b1_1_ba = Parameter(state_dict_pretrain['features.0.bias'], requires_grad=True)     # bias64_1
        self.w1_2_ba = Parameter(state_dict_pretrain['features.2.weight'], requires_grad=True)   # weight64_2
        self.b1_2_ba = Parameter(state_dict_pretrain['features.2.bias'], requires_grad=True)     # bias64_2
        self.w1_1_rg = Parameter(state_dict_pretrain['features.0.weight'], requires_grad=True)  # weight64_1
        self.b1_1_rg = Parameter(state_dict_pretrain['features.0.bias'], requires_grad=True)     # bias64_1
        self.w1_2_rg = Parameter(state_dict_pretrain['features.2.weight'], requires_grad=True)   # weight64_2
        self.b1_2_rg = Parameter(state_dict_pretrain['features.2.bias'], requires_grad=True)     # bias64_2

        '''
        Pre-trained VGG weights - 2nd block
        '''
        self.w2_1_ba = Parameter(state_dict_pretrain['features.5.weight'], requires_grad=True)   # weight128_1
        self.b2_1_ba = Parameter(state_dict_pretrain['features.5.bias'], requires_grad=True)     # bias128_1
        self.w2_2_ba = Parameter(state_dict_pretrain['features.7.weight'], requires_grad=True)   # weight128_2
        self.b2_2_ba = Parameter(state_dict_pretrain['features.7.bias'], requires_grad=True)     # bias128_2
        self.w2_1_rg = Parameter(state_dict_pretrain['features.5.weight'], requires_grad=True)   # weight128_1
        self.b2_1_rg = Parameter(state_dict_pretrain['features.5.bias'], requires_grad=True)     # bias128_1
        self.w2_2_rg = Parameter(state_dict_pretrain['features.7.weight'], requires_grad=True)   # weight128_2
        self.b2_2_rg = Parameter(state_dict_pretrain['features.7.bias'], requires_grad=True)     # bias128_2

        '''
        Pre-trained VGG weights - 3rd block
        '''
        self.w3_1_ba = Parameter(state_dict_pretrain['features.10.weight'], requires_grad=True)  # weight256_1
        self.b3_1_ba = Parameter(state_dict_pretrain['features.10.bias'], requires_grad=True)    # bias256_1
        self.w3_2_ba = Parameter(state_dict_pretrain['features.12.weight'], requires_grad=True)  # weight256_2
        self.b3_2_ba = Parameter(state_dict_pretrain['features.12.bias'], requires_grad=True)    # bias256_2
        self.w3_3_ba = Parameter(state_dict_pretrain['features.14.weight'], requires_grad=True)  # weight256_3
        self.b3_3_ba = Parameter(state_dict_pretrain['features.14.bias'], requires_grad=True)    # bias256_3
        self.w3_1_rg = Parameter(state_dict_pretrain['features.10.weight'], requires_grad=True)  # weight256_1
        self.b3_1_rg = Parameter(state_dict_pretrain['features.10.bias'], requires_grad=True)    # bias256_1
        self.w3_2_rg = Parameter(state_dict_pretrain['features.12.weight'], requires_grad=True)  # weight256_2
        self.b3_2_rg = Parameter(state_dict_pretrain['features.12.bias'], requires_grad=True)    # bias256_2
        self.w3_3_rg = Parameter(state_dict_pretrain['features.14.weight'], requires_grad=True)  # weight256_3
        self.b3_3_rg = Parameter(state_dict_pretrain['features.14.bias'], requires_grad=True)    # bias256_3


        # Branches
        '''Boundary Aware block'''
        self.c1_ba = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=1, kernel_size=(3,3), stride=(1,1), padding=(1,1)),
            nn.ReLU(inplace=True)
        )
        self.c2_ba = nn.Sequential(
            nn.Upsample(size=(112, 112), mode='bilinear'),
            nn.Conv2d(in_channels=128, out_channels=1, kernel_size=(3,3), stride=(1,1), padding=(1,1)),
            nn.ReLU(inplace=True)
        )
        self.c3_ba = nn.Sequential(
            nn.Upsample(size=(112, 112), mode='bilinear'),
            nn.Conv2d(in_channels=256, out_channels=1, kernel_size=(3,3), stride=(1,1), padding=(1,1)),
            nn.ReLU(inplace=True)
        )
        self.c4_ba = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=1, kernel_size=(3,3), stride=(1,1), padding=(1,1)),
        )

        '''Region block'''
        self.c1_rg = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=1, kernel_size=(3,3), stride=(1,1), padding=(1,1)),
            nn.ReLU(inplace=True)
        )
        self.c2_rg = nn.Sequential(
            nn.Upsample(size=(112, 112), mode='bilinear'),
            nn.Conv2d(in_channels=128, out_channels=1, kernel_size=(3,3), stride=(1,1), padding=(1,1)),
            nn.ReLU(inplace=True)
        )
        self.c3_rg = nn.Sequential(
            nn.Upsample(size=(112, 112), mode='bilinear'),
            nn.Conv2d(in_channels=256, out_channels=1, kernel_size=(3,3), stride=(1,1), padding=(1,1)),
            nn.ReLU(inplace=True)
        )
        self.c4_rg = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=1, kernel_size=(3,3), stride=(1,1), padding=(1,1)),
        )

        '''Final combination block'''
        self.c_fin = nn.Sequential(
            nn.Conv2d(in_channels=6, out_channels=64, kernel_size=(3,3), stride=(1,1), padding=(1,1)),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3,3), stride=(1,1), padding=(1,1)),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3,3), stride=(1,1), padding=(1,1)),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=64, out_channels=1, kernel_size=(1,1), stride=(1,1), padding=0),
        )

    def forward(self, x):
        """[1] Define Three Scale-Invariant CNN (Si) for Boundary part"""
        '''1st: keep the kernel by scale=1 -> 3x3, named sz+block+index'''
        out1_sc1_ba, out2_sc1_ba, out3_sc1_ba = self.vgg_forward(x, 1,
                                                                 self.w1_1_ba, self.b1_1_ba, self.w1_2_ba, self.b1_2_ba,
                                                                 self.w2_1_ba, self.b2_1_ba, self.w2_2_ba, self.b2_2_ba,
                                                                 self.w3_1_ba, self.b3_1_ba, self.w3_2_ba, self.b3_2_ba, self.w3_3_ba, self.b3_3_ba)

        '''2nd: expand the kernel by scale=1.666 -> 5x5, named sz+block+index'''
        out1_sc2_ba, out2_sc2_ba, out3_sc2_ba = self.vgg_forward(x, 1.66666666,
                                                                 self.w1_1_ba, self.b1_1_ba, self.w1_2_ba, self.b1_2_ba,
                                                                 self.w2_1_ba, self.b2_1_ba, self.w2_2_ba, self.b2_2_ba,
                                                                 self.w3_1_ba, self.b3_1_ba, self.w3_2_ba, self.b3_2_ba, self.w3_3_ba, self.b3_3_ba)

        '''3rd: expand the kernel by scale=2.333 -> 7x7, named sz+block+index'''
        out1_sc3_ba, out2_sc3_ba, out3_sc3_ba = self.vgg_forward(x, 2.33333333,
                                                                 self.w1_1_ba, self.b1_1_ba, self.w1_2_ba, self.b1_2_ba,
                                                                 self.w2_1_ba, self.b2_1_ba, self.w2_2_ba, self.b2_2_ba,
                                                                 self.w3_1_ba, self.b3_1_ba, self.w3_2_ba, self.b3_2_ba, self.w3_3_ba, self.b3_3_ba)
        
        # Compare vertically (along channel) for 'Boundary' & 'Region'
        out1_em_ba = self.c1_ba(torch.max(out1_sc1_ba, torch.max(out1_sc2_ba, out1_sc3_ba)))
        out2_em_ba = self.c2_ba(torch.max(out2_sc1_ba, torch.max(out2_sc2_ba, out2_sc3_ba)))
        out3_em_ba = self.c3_ba(torch.max(out3_sc1_ba, torch.max(out3_sc2_ba, out3_sc3_ba)))

        # Up-Stream to Boundary output
        '''Boundary branch: upsample & conv to concatenate horizontally'''
        out_cat_ba = torch.cat((out1_em_ba, out2_em_ba, out3_em_ba), 1)
        out_ba = self.c4_ba(out_cat_ba)     # used as input for final predict


        """[2] Define Three Scale-Invariant CNN (Si) for Region part"""
        '''1st: keep the kernel by scale=1 -> 3x3, named sz+block+index'''
        out1_sc1_rg, out2_sc1_rg, out3_sc1_rg = self.vgg_forward(x, 1,
                                                                 self.w1_1_rg, self.b1_1_rg, self.w1_2_rg, self.b1_2_rg,
                                                                 self.w2_1_rg, self.b2_1_rg, self.w2_2_rg, self.b2_2_rg,
                                                                 self.w3_1_rg, self.b3_1_rg, self.w3_2_rg, self.b3_2_rg, self.w3_3_rg, self.b3_3_rg)

        '''2nd: expand the kernel by scale=1.666 -> 5x5, named sz+block+index'''
        out1_sc2_rg, out2_sc2_rg, out3_sc2_rg = self.vgg_forward(x, 1.66666666,
                                                                 self.w1_1_rg, self.b1_1_rg, self.w1_2_rg, self.b1_2_rg,
                                                                 self.w2_1_rg, self.b2_1_rg, self.w2_2_rg, self.b2_2_rg,
                                                                 self.w3_1_rg, self.b3_1_rg, self.w3_2_rg, self.b3_2_rg, self.w3_3_rg, self.b3_3_rg)

        '''3rd: expand the kernel by scale=2.333 -> 7x7, named sz+block+index'''
        out1_sc3_rg, out2_sc3_rg, out3_sc3_rg = self.vgg_forward(x, 2.33333333,
                                                                 self.w1_1_rg, self.b1_1_rg, self.w1_2_rg, self.b1_2_rg,
                                                                 self.w2_1_rg, self.b2_1_rg, self.w2_2_rg, self.b2_2_rg,
                                                                 self.w3_1_rg, self.b3_1_rg, self.w3_2_rg, self.b3_2_rg, self.w3_3_rg, self.b3_3_rg)
        
        # Compare vertically (along channel) for 'Boundary' & 'Region'
        out1_em_rg = self.c1_rg(torch.max(out1_sc1_rg, torch.max(out1_sc2_rg, out1_sc3_rg)))
        out2_em_rg = self.c2_rg(torch.max(out2_sc1_rg, torch.max(out2_sc2_rg, out2_sc3_rg)))
        out3_em_rg = self.c3_rg(torch.max(out3_sc1_rg, torch.max(out3_sc2_rg, out3_sc3_rg)))

        # Bottom-Stream to Region output
        '''Boundary branch: upsample & conv to concatenate horizontally'''
        out_cat_rg = torch.cat((out1_em_rg, out2_em_rg, out3_em_rg), 1)
        out_rg = self.c4_ba(out_cat_rg)     # used as input for final predict


        """[3] Final combination for Boundary and Region part"""
        # Combination Stream to final Region output
        out_cat_fin = torch.cat((out_cat_ba, out_cat_rg), 1)
        out_fin = self.c_fin(out_cat_fin)

        """Activation for all outputs"""
        ba1 = F.sigmoid(out1_em_ba)
        ba2 = F.sigmoid(out2_em_ba)
        ba3 = F.sigmoid(out3_em_ba)
        ba_all = F.sigmoid(out_ba)

        rg1 = F.sigmoid(out1_em_rg)
        rg2 = F.sigmoid(out1_em_rg)
        rg3 = F.sigmoid(out1_em_rg)
        rg_all = F.sigmoid(out_rg)

        fin = F.sigmoid(out_fin)

        return ba_all, ba1, ba2, ba3, rg_all, rg1, rg2, rg3, fin

    def vgg_forward(self, x, scale,
                    w1_1, b1_1, w1_2, b1_2,
                    w2_1, b2_1, w2_2, b2_2,
                    w3_1, b3_1, w3_2, b3_2, w3_3, b3_3):

        # when scale=1, no change on the kernel weights
        if scale == 1:
            # w1_1.requires_grad = True; w1_2.requires_grad = True
            # w2_1.requires_grad = True; w2_2.requires_grad = True
            # w3_1.requires_grad = True; w3_2.requires_grad = True;  w3_3.requires_grad = True

            x = F.conv2d(x, weight=w1_1, bias=b1_1, stride=1, padding=1)
            x = F.leaky_relu(x)
            x = F.conv2d(x, weight=w1_2, bias=b1_2, stride=1, padding=1)
            x_out1 = F.leaky_relu(x)
            x = F.max_pool2d(x_out1, kernel_size=2, stride=2)

            x = F.conv2d(x, weight=w2_1, bias=b2_1, stride=1, padding=1)
            x = F.leaky_relu(x)
            x = F.conv2d(x, weight=w2_2, bias=b2_2, stride=1, padding=1)
            x_out2 = F.leaky_relu(x)
            x = F.max_pool2d(x_out2, kernel_size=2, stride=2)

            x = F.conv2d(x, weight=w3_1, bias=b3_1, stride=1, padding=1)
            x = F.leaky_relu(x)
            x = F.conv2d(x, weight=w3_2, bias=b3_2, stride=1, padding=1)
            x = F.leaky_relu(x)
            x = F.conv2d(x, weight=w3_3, bias=b3_3, stride=1, padding=1)
            x_out3 = F.leaky_relu(x)

        # when scale>1, upsample the kernel weights
        else:
            # w1_1.requires_grad = False; w1_2.requires_grad = False
            # w2_1.requires_grad = False; w2_2.requires_grad = False
            # w3_1.requires_grad = False; w3_2.requires_grad = False;  w3_3.requires_grad = False

            sz1_1 = np.round(scale * np.array(w1_1.size()[2:]))
            sz1_2 = np.round(scale * np.array(w1_2.size()[2:]))
            x = F.conv2d(x, weight=F.upsample(w1_1, size=tuple(sz1_1), mode='bilinear'), bias=b1_1, stride=1, padding=int(sz1_1[0]/2))
            x = F.leaky_relu(x)
            x = F.conv2d(x, weight=F.upsample(w1_2, size=tuple(sz1_2), mode='bilinear'), bias=b1_2, stride=1, padding=int(sz1_2[0]/2))
            x_out1 = F.leaky_relu(x)
            x = F.max_pool2d(x_out1, kernel_size=2, stride=2)

            sz2_1 = np.round(scale * np.array(w2_1.size()[2:]))
            sz2_2 = np.round(scale * np.array(w2_2.size()[2:]))
            x = F.conv2d(x, weight=F.upsample(w2_1, tuple(sz2_1), mode='bilinear'), bias=b2_1, stride=1, padding=int(sz2_1[0]/2))
            x = F.leaky_relu(x)
            x = F.conv2d(x, weight=F.upsample(w2_2, tuple(sz2_2), mode='bilinear'), bias=b2_2, stride=1, padding=int(sz2_2[0]/2))
            x_out2 = F.leaky_relu(x)
            x = F.max_pool2d(x_out2, kernel_size=2, stride=2)

            sz3_1 = np.round(scale * np.array(w3_1.size()[2:]))
            sz3_2 = np.round(scale * np.array(w3_2.size()[2:]))
            sz3_3 = np.round(scale * np.array(w3_3.size()[2:]))
            x = F.conv2d(x, weight=F.upsample(w3_1, tuple(sz3_1), mode='bilinear'), bias=b3_1, stride=1, padding=int(sz3_1[0]/2))
            x = F.leaky_relu(x)
            x = F.conv2d(x, weight=F.upsample(w3_2, tuple(sz3_2), mode='bilinear'), bias=b3_2, stride=1, padding=int(sz3_2[0]/2))
            x = F.leaky_relu(x)
            x = F.conv2d(x, weight=F.upsample(w3_3, tuple(sz3_3), mode='bilinear'), bias=b3_3, stride=1, padding=int(sz3_3[0]/2))
            x_out3 = F.leaky_relu(x)

        return x_out1, x_out2, x_out3


def SiBA_net(fix_para=False, **kwargs):
    """
    Args:
        fix_para (bool): If True, Fix the weights in part of the CNN
    """
    model = SiBANET(**kwargs)

    # Optional: Fix weights in certain layers
    if fix_para is False:
        for para in model.parameters():
            para.requires_grad = True

    return model
