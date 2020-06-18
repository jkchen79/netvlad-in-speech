#! /usr/bin/env python3

import sys
import os
import argparse
import json
import re
import logging

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
import random
from collections import OrderedDict

from options_parser import Options_Parser
from speech_dataloader import SpeechDataLoader
from nnet_model import NNetModel

#  import warnings
#  warnings.filterwarnings('ignore')


def mkdir(dir_path):
    dir_path = dir_path.rstrip('/')
    dirs_to_be_created = []
    while not os.path.exists(dir_path):
        dirs_to_be_created.append(dir_path)
        dir_path = os.path.dirname(dir_path)
    for d in reversed(dirs_to_be_created):
        os.mkdir(d)

def get_logger(log_file):
    log_dir = os.path.dirname(log_file)
    mkdir(log_dir)
    logger = logging.getLogger(log_file)
    logger.setLevel(logging.DEBUG)
    fh = logging.FileHandler(log_file)
    fh.setLevel(logging.DEBUG)
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    ch.setFormatter(formatter)
    fh.setFormatter(formatter)
    logger.addHandler(ch)
    logger.addHandler(fh)
    return logger

def predict(config, logger):
    pass

def do_evaluation(model, dataloader, logger):
    num_samples = 0
    correct_cnt = 0
    int2target = dataloader.get_int2target_dict()
    for i, batch_data in enumerate(dataloader):
        utt = batch_data["utt"]
        feats = torch.unsqueeze(batch_data["feats"], dim=1)
        # transpose the feats to (batch_size, 1, feat_dim, length)
        feats = torch.transpose(feats, 2, 3).cuda()
        targets = batch_data["targets"].cuda()
        # lengths = batch_data["lengths"].cuda()
        num_samples += feats.size(0)
        logits, embd = model(feats)
        pred, acc = model.module.evaluate(logits, targets)
        #  for u, t, p in zip(utt, targets.cpu().numpy(), pred.cpu().numpy()):
        #      logger.info(" ".join((u, int2target[t], int2target[p])))
        correct_cnt += acc 

    acc_rate = float(correct_cnt) / num_samples
    return (acc_rate, num_samples, correct_cnt)

def evaluate(config, logger):
    model = CTC_LID_Model(config)
    ckpt_params = OrderedDict()
    for k, v in torch.load(config.ckpt).items():
        k = re.sub("^module.", "", k)
        ckpt_params[k] = v
    model.load_state_dict(ckpt_params)
    model.eval()
    model = nn.DataParallel(model)
    model = model.cuda()

    eval_dataloader = SpeechDataLoader(config.eval_utt2npy,
                                       utt2target=config.eval_utt2target,
                                       targets_list=config.targets_list,
                                       batch_size=config.batch_size,
                                       num_workers=config.num_workers,
                                       training=False,
                                       shuffle=False,
                                       padding_batch=config.padding_batch
                                       )
    acc_rate, num_samples, correct_cnt = do_evaluation(model, eval_dataloader, logger)
    logger.info("Epoch [%d/%d], eval num_samples = %d, correct_cnt = %d, acc = %.6f" % 
                (epoch, num_epochs, num_samples, correct_cnt, acc_rate))

def setup_optimizer(model, optimizer_name, learning_rate):
    optimizer = None
    if optimizer_name == "sgd":
        optimizer = torch.optim.SGD(model.parameters(), 
                        lr=learning_rate, momentum=0.9, weight_decay=1e-4)
    elif optimizer_name == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    else:
        raise ValueError
    return optimizer

def train(config, logger):
    torch.manual_seed(997)
    mkdir(config.ckpt_dir)

    model = NNetModel(config)

    if config.restore_from_ckpt:
        ckpt_params = OrderedDict()
        for k, v in torch.load(config.ckpt).items():
            k = re.sub("^module.", "", k)
            ckpt_params[k] = v
        model.load_state_dict(ckpt_params)

    model = nn.DataParallel(model)
    model = model.cuda()
    logger.info("nnet model:")
    logger.info(model)

    num_epochs = config.epochs
    learning_rate = config.lr

    criterion = nn.CrossEntropyLoss()
    optimizer = setup_optimizer(model, config.optimizer, learning_rate)
    logger.info(optimizer)

    logger.info("Begin to train ...")
    for epoch in range(num_epochs):
        if epoch == int(num_epochs * 0.4) or epoch == int(num_epochs * 0.75):
            learning_rate /= 10
            logger.info("learning_rate decays to %f" % learning_rate)
            optimizer = setup_optimizer(model, config.optimizer, learning_rate)
            logger.info(optimizer)

        training_dataloader = SpeechDataLoader(
                                       config.train_utt2npy,
                                       utt2target=config.train_utt2target,
                                       targets_list=config.targets_list,
                                       batch_size=config.batch_size,
                                       num_workers=config.num_workers,
                                       training=True,
                                       shuffle=True,
                                       padding_batch=config.padding_batch
                                       )

        dev_dataloader = SpeechDataLoader(config.eval_utt2npy,
                                       utt2target=config.eval_utt2target,
                                       targets_list=config.targets_list,
                                       batch_size=torch.cuda.device_count(),
                                       num_workers=config.num_workers,
                                       training=False,
                                       shuffle=False,
                                       padding_batch=config.padding_batch
                                       )

        num_batchs_per_epoch = training_dataloader.get_dataset_size() / config.batch_size

        for i, batch_data in enumerate(training_dataloader):
            feats = torch.unsqueeze(batch_data["feats"], dim=1)
            # transpose the feats to (batch_size, 1, feat_dim, length)
            feats = torch.transpose(feats, 2, 3).cuda()
            targets = batch_data["targets"].cuda()

            # Forward + Backward + Optimize
            optimizer.zero_grad()
            logits, embd = model(feats)
            loss = criterion(logits, targets)
            loss.backward()
            optimizer.step()

            logger.info("Epoch [%d/%d], Iter [%d/%d], Loss: %.4f, Length [%d]" % 
                    (epoch, num_epochs, i, num_batchs_per_epoch, loss.data, feats.size()[3]))

        if epoch % 10 == 0 or (epoch > num_epochs // 2 and epoch % 5 == 0) or epoch > int(0.9 * num_epochs):
            logger.info("Begin to evaluate dev-set ...")
            acc, num_samples, correct_cnt = do_evaluation(model, dev_dataloader, logger)
            logger.info("Epoch [%d/%d], dev num_samples = %d, correct_cnt = %d, acc = %.6f" % 
                    (epoch, num_epochs, num_samples, correct_cnt, acc))
            ckpt_path = "%s/ckpt-epoch-%d-acc-%.4f.mdl" % (config.ckpt_dir, epoch, acc)
            torch.save(model.state_dict(), ckpt_path)


if __name__ == '__main__':
    description = 'Args Parser.'
    parser = Options_Parser(description)
    config = parser.parse_args()

    logger = get_logger(config.log_file)
    logger.info("config:")
    logger.info(json.dumps(vars(config), indent=4))

    if config.do_train:
        train(config, logger)
    elif config.do_eval:
        evaluate(config, logger)
    elif config.do_predict:
        predict(config, logger)
