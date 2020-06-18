
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class NetVLAD(nn.Module):

    def __init__(self, feature_size, cluster_size, add_batch_norm=True, norm='l2', length_norm=False, use_cuda=True):
        """Initialize a NetVLAD layer.

        Args:
        feature_size:   Dimensionality of the input features.
        cluster_size:   The number of clusters in NetVLAD layer.
        add_batch_norm: If True (default), add the batch normalization when calculating
                        the cluster occupancies.
        """
        super(NetVLAD, self).__init__()
        self.feature_size = feature_size
        self.cluster_size = cluster_size
        self.add_batch_norm = add_batch_norm
        self.norm = norm
        self.length_norm = length_norm
        self.use_cuda = use_cuda

        self.cluster_weight = nn.Parameter(
            torch.Tensor(self.feature_size, self.cluster_size)
                    .normal_(0, 1 / math.sqrt(self.feature_size)))
        self.cluster_bias = nn.Parameter(
            torch.Tensor(self.cluster_size)
                    .normal_(0, 1 / math.sqrt(self.feature_size)))

        self.cluster_mean = nn.Parameter(
            torch.Tensor(1, self.feature_size, self.cluster_size)
                    .normal_(0, 1 / math.sqrt(self.feature_size)))

    @property
    def output_size(self):
        return self.feature_size * self.cluster_size

    def forward(self, input_x):
        """Forward pass of a NetVLAD layer.

        Args:
        input_x: Expected its shape of 'batch_size' x 'feature_size' x 'length',
                 Inputs with its dimension larger than 3 are also accepted,
                 but they will be reshape as 'batch_size' x 'feature_size' x 'L'.
        Returns:
        vlad: the VLAD vector with shape of 'batch_size' x 'cluster_size * feature_size'.
        """
        assert input_x.dim() > 2, 'The input_x.dim must be at least 3'
        x_size = input_x.size()
        assert x_size[1] == self.feature_size, \
            'The second dimensionality of input_x should be the `feature_size`.'
        input_x = input_x.contiguous().view(x_size[0], x_size[1], -1)
        input_x = input_x.transpose(1, 2)

        batch_size, length, feature_size = input_x.size()

        occupancy = torch.matmul(input_x, self.cluster_weight)
        if self.add_batch_norm:
            self.batch_norm = nn.BatchNorm1d(length)
            if self.use_cuda:
                self.batch_norm = self.batch_norm.cuda()
            occupancy = self.batch_norm(occupancy)
        else:
            occupancy = torch.add(occupancy, self.cluster_bias)

        occupancy = nn.functional.softmax(occupancy, dim=-1)

        #  vlad[k] = sum_i(occ_ik * (x_i - mean_k))
        #          = sum_i(occ_ik * x_i) - mean_k * sum_i(occ_ik)
        #

        cluster_occ_sum = torch.sum(occupancy, 1, keepdim=True)
        cluster_mean_occ_sum = torch.mul(cluster_occ_sum, self.cluster_mean)

        occupancy = torch.transpose(occupancy, 2, 1)

        vlad = torch.matmul(occupancy, input_x)
        vlad = torch.transpose(vlad, 2, 1)
        vlad = vlad - cluster_mean_occ_sum
        # the shape of vald is: batch_size x feature_size x cluster_size

        if self.length_norm:
            vlad = vlad.contiguous().view(batch_size, -1)
            vlad = torch.div(vlad, length)

        if 'l2' == self.norm:
            # column-wised normalization (intra-normalization).
            # the dim option of F.normalize() must be the feature_size
            vlad = nn.functional.normalize(vlad, p=2, dim=1)
            vlad = vlad.permute(0, 2, 1)
            vlad = vlad.contiguous().view(batch_size, -1)
            vlad = F.normalize(vlad, dim=1)
        elif 'mass_norm' == self.norm:
            #  cluster_occ_sum = cluster_occ_sum.squeeze().contiguous().view(
                #  batch_size, self.cluster_size, 1)
            cluster_occ_sum = cluster_occ_sum.permute(0, 2, 1)
            vlad = vlad.permute(0, 2, 1)
            vlad = torch.div(vlad, cluster_occ_sum)
            vlad = vlad.contiguous().view(batch_size, -1)

        return vlad
