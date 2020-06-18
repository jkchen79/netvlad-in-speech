#! /usr/bin/env python3

# Copyright 2018  Sun Yat-sen University (author: Jinkun Chen)

import sys
import os
import argparse
from collections import OrderedDict
import numpy as np

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

from modules.resnet import ResidualBlock, ResNet
from modules.avgpool import AvgPool
from modules.netvlad import NetVLAD


class NNetModel (nn.Module):

    def __init__(self, config, use_cuda=True):
        super(NNetModel, self).__init__()
        self.config = config
        self.use_cuda = use_cuda
        self.bn_size = config.bn_size
        self.l2norm_constrained = config.l2norm_constrained

        resnet_layers = {'resnet18': [2, 2, 2, 2], 'resnet34': [3, 4, 6, 3]}
        self.resnet = ResNet(ResidualBlock, resnet_layers[config.resnet])
        resnet_feats_size = 128

        #  self.dropout = nn.Dropout2d(p=config.dropout_rate)

        if 'avgpool' == config.encoder:
            self.encoder = AvgPool(resnet_feats_size, config.avgpool_norm)
        elif 'netvlad' == config.encoder:
            self.encoder = NetVLAD(resnet_feats_size, config.cluster_size)
        else:
            raise KeyError("%s is not supported!" % config.encoder)
        encoder_out_size = self.encoder.output_size

        self.fc1_layer = nn.Linear(encoder_out_size, config.bn_size)
        self.fc2_layer = nn.Linear(config.bn_size, config.num_classes)
        if self.l2norm_constrained:
            if fixed_l2norm_alpha > 0.0:
                self.l2norm_alpha = config.fixed_l2norm_alpha
            else:
                self.l2norm_alpha = nn.Parameter(torch.Tensor(1).set_(
                    torch.FloatTensor([config.initial_l2norm_alpha])), requires_grad=True)

    def forward(self, x):
        cnn_out = self.resnet(x)
        #  if self.config.do_train:
        #      cnn_out = self.dropout(cnn_out)
        encoder_out = self.encoder(cnn_out)

        embd = self.fc1_layer(encoder_out)
        embd = F.normalize(embd, p=2, dim=1)
        if self.l2norm_constrained:
            embd = torch.mul(embd, self.l2norm_alpha)
        out = self.fc2_layer(embd)
        return out, embd

    def evaluate(self, logits, targets):
        _, pred = logits.max(dim=1)
        correct = pred.eq(targets).float()
        acc = correct.sum().item() 
        return (pred, acc)
