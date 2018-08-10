import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import torch.utils.model_zoo as model_zoo
import torchvision
from collections import OrderedDict

from PIL import Image
import os
import os.path
import numpy as np
import math
import ipdb


""" WSSS-Net """

class VGG16features(nn.Module):
    def __init__(self, activation=F.relu):
        super(VGG16features, self).__init__()

        self.activation = activation
        self.conv1_1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=(33, 33))
        self.conv1_2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        self.conv2_1 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.conv2_2 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        self.conv3_1 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.conv3_2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.conv3_3 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        self.conv4_1 = nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1)
        self.conv4_2 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.conv4_3 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        self.conv5_1 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.conv5_2 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.conv5_3 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.pool5 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, (2. / n) ** .5)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        x = self.conv1_1(x)
        x = self.activation(x)
        x = self.conv1_2(x)
        c1 = self.activation(x)

        x = self.pool1(c1)

        x = self.conv2_1(x)
        x = self.activation(x)
        x = self.conv2_2(x)
        c2 = self.activation(x)

        x = self.pool2(c2)

        x = self.conv3_1(x)
        x = self.activation(x)
        x = self.conv3_2(x)
        x = self.activation(x)
        x = self.conv3_3(x)
        c3 = self.activation(x)

        x = self.pool3(c3)

        x = self.conv4_1(x)
        x = self.activation(x)
        x = self.conv4_2(x)
        x = self.activation(x)
        x = self.conv4_3(x)
        c4 = self.activation(x)

        x = self.pool4(c4)

        x = self.conv5_1(x)
        x = self.activation(x)
        x = self.conv5_2(x)
        x = self.activation(x)
        x = self.conv5_3(x)
        c5 = self.activation(x)

        x = self.pool5(c5)

        return x

    def forward_hypercol(self, x):
        x = self.conv1_1(x)
        x = self.activation(x)
        x = self.conv1_2(x)
        c1 = self.activation(x)

        x = self.pool1(c1)

        x = self.conv2_1(x)
        x = self.activation(x)
        x = self.conv2_2(x)
        c2 = self.activation(x)

        x = self.pool2(c2)

        x = self.conv3_1(x)
        x = self.activation(x)
        x = self.conv3_2(x)
        x = self.activation(x)
        x = self.conv3_3(x)
        c3 = self.activation(x)

        x = self.pool3(c3)

        x = self.conv4_1(x)
        x = self.activation(x)
        x = self.conv4_2(x)
        x = self.activation(x)
        x = self.conv4_3(x)
        c4 = self.activation(x)

        x = self.pool4(c4)

        x = self.conv5_1(x)
        x = self.activation(x)
        x = self.conv5_2(x)
        x = self.activation(x)
        x = self.conv5_3(x)
        c5 = self.activation(x)

        return c1, c2, c3, c4, c5

# VGG model definition
class VGG(nn.Module):

    def __init__(self, features, num_classes=1000):
        super(VGG, self).__init__()
        self.features = features
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, num_classes),
        )
        self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, (2. / n)**.5)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()

# load pre-trained VGG-16 model
def vgg16(pretrained=False, **kwargs):
    """VGG 16-layer model (configuration "D")
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    VGG16fs = VGG16features()
    model = VGG(VGG16fs, **kwargs)
    if pretrained:
        state_dict = torch.utils.model_zoo.load_url(torchvision.models.vgg.model_urls['vgg16'])
        new_state_dict = {}

        original_layer_ids = set()
        # copy the classifier entries and make a mapping for the feature mappings
        for key in state_dict.keys():
            if 'classifier' in key:
                new_state_dict[key] = state_dict[key]
            elif 'features' in key:
                original_layer_ids.add(int(key.split('.')[1]))
        sorted_original_layer_ids = sorted(list(original_layer_ids))

        layer_ids = set()
        for key in model.state_dict().keys():
            if 'classifier' in key:
                continue
            elif 'features' in key:
                layer_id = key.split('.')[1]
                layer_ids.add(layer_id)
        sorted_layer_ids = sorted(list(layer_ids))

        for key, value in state_dict.items():
            if 'features' in key:
                original_layer_id = int(key.split('.')[1])
                original_param_id = key.split('.')[2]
                idx = sorted_original_layer_ids.index(original_layer_id)
                new_layer_id = sorted_layer_ids[idx]
                new_key = 'features.' + new_layer_id + '.' + original_param_id
                new_state_dict[new_key] = value

        model.load_state_dict(new_state_dict)
    return model, VGG16fs

# definition of WSSS module
class WSSSNet(nn.Module):
    def __init__(self, pretrained=True):
        # define VGG architecture and layers
        super(WSSSNet, self).__init__()
        
        '''Define the appearance HNN for binary mask'''
        # define fully-convolutional layers
        self.dsn1A = nn.Conv2d(64, 1, 1)
        self.dsn2A = nn.Conv2d(128, 1, 1)
        self.dsn3A = nn.Conv2d(256, 1, 1)
        self.dsn4A = nn.Conv2d(512, 1, 1)
        self.dsn5A = nn.Conv2d(512, 1, 1)
        self.dsn6A = nn.Conv2d(5, 1, 1)
        
        # define upsampling/deconvolutional layers
        self.upscore2A = nn.Upsample(scale_factor=2, mode='bilinear')
        self.upscore3A = nn.Upsample(scale_factor=4, mode='bilinear')
        self.upscore4A = nn.Upsample(scale_factor=8, mode='bilinear')
        self.upscore5A = nn.Upsample(scale_factor=16, mode='bilinear')


        '''Define the edge HNN for binary edge'''
        # define fully-convolutional layers
        self.dsn1E = nn.Conv2d(64, 1, 1)
        self.dsn2E = nn.Conv2d(128, 1, 1)
        self.dsn3E = nn.Conv2d(256, 1, 1)
        self.dsn4E = nn.Conv2d(512, 1, 1)
        self.dsn5E = nn.Conv2d(512, 1, 1)
        self.dsn6E = nn.Conv2d(5, 1, 1)
        
        # define upsampling/deconvolutional layers
        self.upscore2E = nn.Upsample(scale_factor=2, mode='bilinear')
        self.upscore3E = nn.Upsample(scale_factor=4, mode='bilinear')
        self.upscore4E = nn.Upsample(scale_factor=8, mode='bilinear')
        self.upscore5E = nn.Upsample(scale_factor=16, mode='bilinear')

        
        # initialize weights of layers
        for m in self.named_modules():
            if m[0] == 'dsn6A' or m[0] == 'dsn6E':
                m[1].weight.data.fill_(0.2)
            elif isinstance(m[1], nn.Conv2d):
                n = m[1].kernel_size[0] * m[1].kernel_size[1] * m[1].out_channels
                m[1].weight.data.normal_(0, (2. / n) ** .5)
            elif isinstance(m[1], nn.BatchNorm2d):
                m[1].weight.data.fill_(1)
                m[1].bias.data.zero_()

        
        _, VGG16fsA = vgg16(pretrained=pretrained)
        self.VGG16fsA = VGG16fsA

        _, VGG16fsE = vgg16(pretrained=pretrained)
        self.VGG16fsE = VGG16fsE

    # define the computation graph
    def forward(self, x):

        size = x.size()[2:4]
        
        '''Appearance HNN'''
        # get output from VGG model
        conv1A, conv2A, conv3A, conv4A, conv5A = self.VGG16fsA.forward_hypercol(x)

        ## side output
        dsn5_upA = self.upscore5A(self.dsn5A(conv5A))
        d5A = self.crop(dsn5_upA, size)
        
        dsn4_upA = self.upscore4A(self.dsn4A(conv4A))
        d4A = self.crop(dsn4_upA, size)
        
        dsn3_upA = self.upscore3A(self.dsn3A(conv3A))
        d3A = self.crop(dsn3_upA, size)
        
        dsn2_upA = self.upscore2A(self.dsn2A(conv2A))
        d2A = self.crop(dsn2_upA, size)
        
        dsn1A = self.dsn1A(conv1A)
        d1A = self.crop(dsn1A, size)

        # weighted fusion (with learning fusion weights)
        d6A = self.dsn6A(torch.cat((d1A, d2A, d3A, d4A, d5A), 1))
        
        d1A = F.sigmoid(d1A)
        d2A = F.sigmoid(d2A)
        d3A = F.sigmoid(d3A)
        d4A = F.sigmoid(d4A)
        d5A = F.sigmoid(d5A)
        d6A = F.sigmoid(d6A)


        '''Edge HNN'''
        # get output from VGG model
        conv1E, conv2E, conv3E, conv4E, conv5E = self.VGG16fsE.forward_hypercol(x)

        ## side output
        dsn5_upE = self.upscore5E(self.dsn5E(conv5E))
        d5E = self.crop(dsn5_upE, size)
        
        dsn4_upE = self.upscore4E(self.dsn4E(conv4E))
        d4E = self.crop(dsn4_upE, size)
        
        dsn3_upE = self.upscore3E(self.dsn3E(conv3E))
        d3E = self.crop(dsn3_upE, size)
        
        dsn2_upE = self.upscore2E(self.dsn2E(conv2E))
        d2E = self.crop(dsn2_upE, size)
        
        dsn1E = self.dsn1E(conv1E)
        d1E = self.crop(dsn1E, size)

        # weighted fusion (with learning fusion weights)
        d6E = self.dsn6E(torch.cat((d1E, d2E, d3E, d4E, d5E), 1))
        
        d1E = F.sigmoid(d1E)
        d2E = F.sigmoid(d2E)
        d3E = F.sigmoid(d3E)
        d4E = F.sigmoid(d4E)
        d5E = F.sigmoid(d5E)
        d6E = F.sigmoid(d6E)


        return d1A, d2A, d3A, d4A, d5A, d6A, d1E, d2E, d3E, d4E, d5E, d6E
    
    # function to crop the padding pixels
    def crop(self, d, size):
        d_h, d_w = d.size()[2:4]
        g_h, g_w = size[0], size[1]
        d1 = d[:, :, int(math.floor((d_h - g_h)/2.0)):int(math.floor((d_h - g_h)/2.0)) + g_h, int(math.floor((d_w - g_w)/2.0)):int(math.floor((d_w - g_w)/2.0)) + g_w]
        return d1