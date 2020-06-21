#! /usr/bin/env python3

# Copyright 2018  Sun Yat-sen University (author: Jinkun Chen)

import sys
import argparse
import json
import time
import os

class Options_Parser:

    def __init__(self, description=""):
        self.parser = argparse.ArgumentParser(description=description)
        self._dataset_options()
        self._dataloader_options()
        self._model_options()
        self._optimizer_options()
        self._running_options()

    def _dataset_options(self):
        self.parser.add_argument(
            "--train-utt2npy", dest="train_utt2npy", type=str, default=None,
            help="The path of the utt2npy file for training.")
        self.parser.add_argument(
            "--train-utt2target", dest="train_utt2target", type=str, default=None,
            help="The path of the utt2target file for training.") 
        self.parser.add_argument(
            "--eval-utt2npy", dest="eval_utt2npy", type=str, default=None,
            help="The path of the utt2npy file for evaluation.")
        self.parser.add_argument(
            "--eval-utt2target", dest="eval_utt2target", type=str, default=None,
            help="The path of the utt2target file for evaluation.")
        self.parser.add_argument(
            "--targets-list", dest="targets_list", type=str, default=None,
            help="The path of the targets_list file.")

    def _dataloader_options(self):
        self.parser.add_argument(
            '--epochs', dest='epochs', type=int, default=90,
            help='The maximum epochs for training phase.')
        self.parser.add_argument(
            '--batch-size', dest='batch_size', type=int, default=64,
            help='Batch size for DataLoader in training phase.')
        self.parser.add_argument(
            '--fixed-length', dest='fixed_length', type=int, default=0,
            help='If fixed_length > 0, the data loader generates feature sequences in fixed length.')
        self.parser.add_argument(
            '--truncated-min-len', dest='truncated_min_len', type=int, default=300,
            help='the lower bound of the variable truncated range.')
        self.parser.add_argument(
            '--truncated-max-len', dest='truncated_max_len', type=int, default=1024,
            help='the upper bound of the variable truncated range.')
        self.parser.add_argument(
            '--padding-batch', dest='padding_batch', action='store_true',
            help='If True, pad sequence to the max length in a batch.')
        self.parser.add_argument(
            '--num-workers', dest='num_workers', type=int, default=10,
            help='the number of the workers to load data parallelly.')

    def _model_options(self):
        self.parser.add_argument(
            '--feat-dim', dest='feat_dim', type=int, default=40,
            help='The dimension of acustic features.')
        self.parser.add_argument(
            '--hidden-size', dest='hidden_size', type=int, default=1024,
            help='The hidden size of the recurrent layers.')
        self.parser.add_argument(
            '--num-rnn-layers', dest='num_rnn_layers', type=int, default=1,
            help='The number of recurrent layers.')
        self.parser.add_argument(
            '--bidirectional', dest='bidirectional', action='store_true',
            help='If True, use a bidirectional recurrent layer.')
        self.parser.add_argument(
            '--bn-size', dest='bn_size', type=int, default=256,
            help='The size of bottleneck representation.')
        self.parser.add_argument(
            '--num-classes', dest='num_classes', type=int, default=10,
            help='The number of classes (targets).')

        self.parser.add_argument(
            '--resnet', dest='resnet', type=str, default='resnet34',
            help='Type of encoder, avgpool, netvlad ...')

        self.parser.add_argument(
            '--dropout-rate', dest='dropout_rate', type=float, default=0.2,
            help='The dropout rate for CNN layers output.')

        self.parser.add_argument(
            '--encoder', dest='encoder', type=str, default='avgpool',
            help='Type of encoder, avgpool, netvlad ...')
        self.parser.add_argument(
            '--cluster-size', dest='cluster_size', type=int, default=64,
            help='The cluster_size of encoder (Except for avgpool).')
        self.parser.add_argument(
            '--avgpool-norm', dest='avgpool_norm', action='store_true',
            help='If True, outputs of the AvgPool layer will be L2-normalized.')

        self.parser.add_argument(
            '--l2norm-constrained', dest='l2norm_constrained',  action='store_true',
            help='If True, multiple a learnable l2norm-alpha after L2Norm layers (default is False).')
        self.parser.add_argument(
            '--fixed-l2norm-alpha', dest='fixed_l2norm_alpha', type=float, default=0.0,
            help='Fixed L2Norm alpha for L2Norm-constrained')
        self.parser.add_argument(
            '--initial-l2norm-alpha', dest='initial_l2norm_alpha', type=float, default=5.0,
            help='Initial L2Norm alpha of the learnable l2norm-alpha ')

    def _optimizer_options(self):
        self.parser.add_argument(
            "--optimizer", dest="optimizer", type=str, default="sgd",
            help="The type of optimizer.")
        self.parser.add_argument(
            "--lr", dest="lr", type=float, default=0.1,
            help="The initial learning rate.")

    def _running_options(self):
        self.parser.add_argument(
            "--ckpt-dir", dest="ckpt_dir", type=str, default="../ckpt",
            help="the directory to save checkpoint files.")
        self.parser.add_argument(
            "--ckpt", dest="ckpt", type=str, default="",
            help="the path of the checkpoint file to be restored.")
        self.parser.add_argument(
            '--do-train', dest='do_train', action='store_true',
            help='If True, run the CTC-LID model in training mode.')
        self.parser.add_argument(
            '--do-eval', dest='do_eval', action='store_true',
            help='If True, run the CTC-LID model in evaluation mode.')
        self.parser.add_argument(
            '--do-predict', dest='do_predict', action='store_true',
            help='If True, run the CTC-LID model in prediction mode.')
        self.parser.add_argument(
            '--restore-from-ckpt', dest='restore_from_ckpt', action='store_true',
            help='If True, restore the model from a checkpont file.')
        self.parser.add_argument(
            "--log-file", dest="log_file", type=str, default="",
            help="The path of logging file for the running job.")

    def parse_args(self):
        config = self.parser.parse_args()
        mode = ""
        if config.do_train:
            mode = 'train'
        elif config.do_eval:
            mode = 'test'
        elif config.do_predict:
            mode = 'predict'

        timestamp = time.strftime("%m%d-%H%M%S", time.localtime())
        config.ckpt_dir = config.ckpt_dir.rstrip('/')
        if config.do_train and os.path.basename(config.ckpt_dir) == 'ckpt':
            config.ckpt_dir = os.path.join(config.ckpt_dir, "job_%s_%s" % (mode, timestamp))

        if len(mode) > 0 and len(config.log_file) == 0:
            config.log_file = "./log/job_%s_model_%s.log.txt" % (mode, timestamp)

        if config.restore_from_ckpt or config.do_eval or config.do_predict:
            assert os.path.isfile(config.ckpt) and os.path.exists(config.ckpt)

        if config.do_eval or config.do_predict:
            config.dropout_rate = 0
            config.batch_size = 1

        return config

    def register(self, arg, dest, arg_type, default=None, action=None, help=""):
        if isinstance(arg_type, bool) and action is not None:
            self.parser.add_argument(
                arg, dest=dest, action=action, help=help)
        else:
            self.parser.add_argument(
                arg, dest=dest, type=arg_type, default=default, help=help)


if __name__ == '__main__':
    parser = Options_Parser("debug")
    args = parser.parse_args()
    print(json.dumps(vars(args), indent=4))
