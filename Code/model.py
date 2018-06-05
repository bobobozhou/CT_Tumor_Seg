import torch
import torch.nn as nn
import torch.nn.functional as F
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


""" Scale-invariant Boundary Aware Net (SiBA-Net) """


class SiBANET(nn.Module):
    def __init__(self, num_classes=4):
        super(SiBANET, self).__init__()

        # TODO: load pre-trained weights from VGG16-Net
        # load parts of pre-train
        state_dict_pretrain = model_zoo.load_url(model_urls['vgg16'])

        '''
        Pre-trained VGG weights - 1st block
        '''
        self.w1_1 = nn.Parameter(state_dict_pretrain['features.0.weight'])  # weight64_1
        self.b1_1 = nn.Parameter(state_dict_pretrain['features.0.bias'])     # bias64_1
        self.w1_2 = nn.Parameter(state_dict_pretrain['features.2.weight'])   # weight64_2
        self.b1_2 = nn.Parameter(state_dict_pretrain['features.2.bias'])     # bias64_2

        '''
        Pre-trained VGG weights - 2nd block
        '''
        self.w2_1 = nn.Parameter(state_dict_pretrain['features.5.weight'])   # weight128_1
        self.b2_1 = nn.Parameter(state_dict_pretrain['features.5.bias'])     # bias128_1
        self.w2_2 = nn.Parameter(state_dict_pretrain['features.7.weight'])   # weight128_2
        self.b2_2 = nn.Parameter(state_dict_pretrain['features.7.bias'])     # bias128_2

        '''
        Pre-trained VGG weights - 3rd block
        '''
        self.w3_1 = nn.Parameter(state_dict_pretrain['features.10.weight'])  # weight256_1
        self.b3_1 = nn.Parameter(state_dict_pretrain['features.10.bias'])    # bias256_1
        self.w3_2 = nn.Parameter(state_dict_pretrain['features.12.weight'])  # weight256_2
        self.b3_2 = nn.Parameter(state_dict_pretrain['features.12.bias'])    # bias256_2
        self.w3_3 = nn.Parameter(state_dict_pretrain['features.14.weight'])  # weight256_3
        self.b3_3 = nn.Parameter(state_dict_pretrain['features.14.bias'])    # bias256_3

        # Branches
        '''Boundary Aware block'''
        self.c1_ba = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3,3), stride=(1,1), padding=(2,2)),
            nn.ReLU(inplace=True)
        )
        self.c2_ba = nn.Sequential(
            nn.Upsample(size=(112, 112), mode='bilinear'),
            nn.Conv2d(in_channels=128, out_channels=64, kernel_size=(3, 3), stride=(1, 1), padding=(2, 2)),
            nn.ReLU(inplace=True)
        )
        self.c3_ba = nn.Sequential(
            nn.Upsample(size=(112, 112), mode='bilinear'),
            nn.Conv2d(in_channels=256, out_channels=64, kernel_size=(3, 3), stride=(1, 1), padding=(2, 2)),
            nn.ReLU(inplace=True)
        )
        self.c4_ba = nn.Conv2d(in_channels=192, out_channels=2, kernel_size=(1, 1), stride=(1, 1), padding=(2, 2))

        '''Region block'''
        self.c1_rg = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), stride=(1, 1), padding=(2, 2)),
            nn.ReLU(inplace=True)
        )
        self.c2_rg = nn.Sequential(
            nn.Upsample(size=(112, 112), mode='bilinear'),
            nn.Conv2d(in_channels=128, out_channels=64, kernel_size=(3, 3), stride=(1, 1), padding=(2, 2)),
            nn.ReLU(inplace=True)
        )
        self.c3_rg = nn.Sequential(
            nn.Upsample(size=(112, 112), mode='bilinear'),
            nn.Conv2d(in_channels=256, out_channels=64, kernel_size=(3, 3), stride=(1, 1), padding=(2, 2)),
            nn.ReLU(inplace=True)
        )
        self.c4_rg = nn.Conv2d(in_channels=192, out_channels=2, kernel_size=(1, 1), stride=(1, 1), padding=(2, 2))

        '''Final combination block'''
        self.c_fin = nn.Sequential(
            nn.Conv2d(in_channels=4, out_channels=64, kernel_size=(3, 3), stride=(1, 1), padding=(2, 2)),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), stride=(1, 1), padding=(2, 2)),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), stride=(1, 1), padding=(2, 2)),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=64, out_channels=2, kernel_size=(3, 3), stride=(1, 1), padding=(2, 2)),
        )

    def forward(self, x):
        # Define Three Scale-Invariant CNN (Si)
        '''1st: keep the kernel by scale=1, named sz+block+index'''
        out1_sc1, out2_sc1, out3_sc1 = self.vgg_forward(x, 1,
                                                        self.w1_1, self.b1_1, self.w1_2, self.b1_2,
                                                        self.w2_1, self.b2_1, self.w2_2, self.b2_2,
                                                        self.w3_1, self.b3_1, self.w3_2, self.b3_2, self.w3_3, self.b3_3)

        '''2nd: expand the kernel by scale=2, named sz+block+index'''
        out1_sc2, out2_sc2, out3_sc2 = self.vgg_forward(x, 2,
                                                        self.w1_1, self.b1_1, self.w1_2, self.b1_2,
                                                        self.w2_1, self.b2_1, self.w2_2, self.b2_2,
                                                        self.w3_1, self.b3_1, self.w3_2, self.b3_2, self.w3_3, self.b3_3)

        '''3rd: expand the kernel by scale=3, named sz+block+index'''
        out1_sc3, out2_sc3, out3_sc3 = self.vgg_forward(x, 3,
                                                        self.w1_1, self.b1_1, self.w1_2, self.b1_2,
                                                        self.w2_1, self.b2_1, self.w2_2, self.b2_2,
                                                        self.w3_1, self.b3_1, self.w3_2, self.b3_2, self.w3_3, self.b3_3)

        # Concatenate vertically for 'Boundary' & 'Region'
        out1_cat = torch.cat((out1_sc1, out1_sc2, out1_sc3), 0)
        out2_cat = torch.cat((out2_sc1, out2_sc2, out2_sc3), 0)
        out3_cat = torch.cat((out3_sc1, out3_sc2, out3_sc3), 0)

        # Up-Stream to Boundary output
        '''Boundary branch: upsample & conv to concatenate horizontally'''
        out1_cat_ba = self.c1_ba(out1_cat)
        out2_cat_ba = self.c2_ba(out2_cat)
        out3_cat_ba = self.c3_ba(out3_cat)
        out_cat_ba = torch.cat((out1_cat_ba, out2_cat_ba, out3_cat_ba), 0)
        out_ba = self.c4_ba(out_cat_ba)     # used as input for final predict

        # Bottom-Stream to Region output
        '''Regions branch: upsample & conv to concatenate horizontally'''
        out1_cat_rg = self.c1_rg(out1_cat)
        out2_cat_rg = self.c2_rg(out2_cat)
        out3_cat_rg = self.c3_rg(out3_cat)
        out_cat_rg = torch.cat((out1_cat_rg, out2_cat_rg, out3_cat_rg), 0)
        out_rg = self.c4_rg(out_cat_rg)     # used as input for final predict

        # Combination Stream to final Region output
        out_cat_fin = torch.cat((out_cat_ba, out_rg), 0)
        out_fin = self.c_fin(out_cat_fin)

        return out_ba, out_rg, out_fin

    def vgg_forward(self, x, scale,
                    w1_1, b1_1, w1_2, b1_2,
                    w2_1, b2_1, w2_2, b2_2,
                    w3_1, b3_1, w3_2, b3_2, w3_3, b3_3):

        sz1_1 = np.concatenate((np.array(w1_1.size()[:2]),
                                scale * np.array(w1_1.size()[2:])), axis=0)
        sz1_2 = np.concatenate((np.array(self.w1_2.size()[:2]),
                                scale * np.array(self.w1_2.size()[2:])), axis=0)
        x = F.conv2d(x, weight=w1_1.resize_(sz1_1), bias=b1_1, stride=1, padding=2)
        x = F.leaky_relu(x)
        x = F.conv2d(x, weight=w1_2.resize_(sz1_2), bias=b1_2, stride=1, padding=2)
        x = F.leaky_relu(x)
        x_out1 = F.max_pool2d(x, kernel_size=2, stride=2)

        sz2_1 = np.concatenate((np.array(w2_1.size()[:2]),
                                scale * np.array(w2_1.size()[2:])), axis=0)
        sz2_2 = np.concatenate((np.array(w2_2.size()[:2]),
                                scale * np.array(w2_2.size()[2:])), axis=0)
        x = F.conv2d(x_out1, weight=w2_1.resize_(sz2_1), bias=b2_1, stride=1, padding=2)
        x = F.leaky_relu(x)
        x = F.conv2d(x, weight=w2_2.resize_(sz2_2), bias=b2_2, stride=1, padding=2)
        x = F.leaky_relu(x)
        x_out2 = F.max_pool2d(x, kernel_size=2, stride=2)

        sz3_1 = np.concatenate((np.array(w3_1.size()[:2]),
                                scale * np.array(w3_1.size()[2:])), axis=0)
        sz3_2 = np.concatenate((np.array(w3_2.size()[:2]),
                                scale * np.array(w3_2.size()[2:])), axis=0)
        sz3_3 = np.concatenate((np.array(w3_3.size()[:2]),
                                scale * np.array(w3_3.size()[2:])), axis=0)
        x = F.conv2d(x_out2, weight=w3_1.resize_(sz3_1), bias=b3_1, stride=1, padding=2)
        x = F.leaky_relu(x)
        x = F.conv2d(x, weight=w3_2.resize_(sz3_2), bias=b3_2, stride=1, padding=2)
        x = F.leaky_relu(x)
        x = F.conv2d(x, weight=w3_3.resize_(sz3_3), bias=b3_3, stride=1, padding=2)
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
