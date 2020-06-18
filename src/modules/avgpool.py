#! /usr/bin/env python3

# Copyright 2018  Sun Yat-sen University (author: Jinkun Chen)

import torch
import torch.nn as nn
import torch.nn.functional as F


class AvgPool(nn.Module):

    def __init__(self, feature_size, l2norm=False):
        super(AvgPool, self).__init__()
        self.l2norm = l2norm
        self.output_size = feature_size

    def forward(self, x):
        batch_size, channels, d_size, lenght = x.size()
        x_mean = F.avg_pool2d(x, (d_size, lenght))
        x_mean = x_mean.view(batch_size, -1)
        if self.l2norm:
            x_mean = F.normalize(x_mean, dim=1)
        out = x_mean
        return out

    @property
    def encoder_out_size(self):
        return self.output_size
