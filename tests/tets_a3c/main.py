# -*- coding: utf-8 -*-
import argparse
import os
import sys

import torch
import torch.optim as optim
import torch.multiprocessing as mp
import torch.nn as nn
import torch.nn.functional as F
from cfrl.wrappers.atari_wrappers import make_atari, wrap_deepmind
from cfrl.nn.atari import AtariLSTMNet, AtariNet
from cfrl.optimizers.shared_rmsprop import SharedRMSprop
from cfrl.optimizers.shared_adam import SharedAdam
import time
from test_a3c import test, train


parser = argparse.ArgumentParser(description='A3C')
parser.add_argument('--lr', type=float, default=0.0001, metavar='LR',
                    help='learning rate (default: 0.0001)')
parser.add_argument('--gamma', type=float, default=0.99, metavar='G',
                    help='discount factor for rewards (default: 0.99)')
parser.add_argument('--tau', type=float, default=1.00, metavar='T',
                    help='parameter for GAE (default: 1.00)')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--num-processes', type=int, default=16, metavar='NP',
                    help='how many training processes to use (default: 16)')
parser.add_argument('--num-steps', type=int, default=20, metavar='NS',
                    help='number of forward steps in A3C (default: 20)')
parser.add_argument('--max-episode-length', type=int, default=10000, metavar='M',
                    help='maximum length of an episode (default: 10000)')
parser.add_argument('--env-name', default='Pong-v0', metavar='ENV',
                    help='environment to train on (default: Pong-v0)')
parser.add_argument('--env-config', default='config.json', metavar='EC',
                    help='environment to crop and resize info (default: config.json)')
parser.add_argument('--shared-optimizer', default=True, metavar='SO',
                    help='use an optimizer without shared statistics.')
parser.add_argument('--load', default=False, metavar='L',
                    help='load a trained model')
parser.add_argument('--save-score-level', type=int, default=20, metavar='SSL',
                    help='reward score test evaluation must get higher than to save model')
parser.add_argument('--optimizer', default='Adam', metavar='OPT',
                    help='shares optimizer choice of Adam or RMSprop')
parser.add_argument('--count-lives', default=True, metavar='CL',
                    help='end of life is end of training episode.')
parser.add_argument('--load-model-dir', default='trained_models/', metavar='LMD',
                    help='folder to load trained models from')
parser.add_argument('--save-model-dir', default='trained_models/', metavar='SMD',
                    help='folder to save trained models')
parser.add_argument('--log-dir', default='logs/', metavar='LG',
                    help='folder to save logs')

if __name__ == '__main__':
    args = parser.parse_args()
    torch.set_default_tensor_type('torch.FloatTensor')
    torch.manual_seed(args.seed)
    
    env = wrap_deepmind(make_atari(args.env_name))
    shared_model = AtariNet(env.observation_space.shape[0], env.action_space)
    if args.load:
        saved_state = torch.load('{0}{1}.dat'.format(args.load_model_dir, args.env_name))
        shared_model.load_state_dict(saved_state)
    shared_model.share_memory()

    if args.shared_optimizer:
        if args.optimizer == 'RMSprop':
            optimizer = SharedRMSprop(
                shared_model.parameters(), lr=args.lr)
        if args.optimizer == 'Adam':
            optimizer = SharedAdam(
                shared_model.parameters(), lr=args.lr)
        optimizer.share_memory()
    else:
    	optimizer = None

    processes = []
    p = mp.Process(target=test, args=(
        args.num_processes, args, shared_model))
    p.start()
    processes.append(p)
    time.sleep(0.1)
    for rank in range(0, args.num_processes):
        p = mp.Process(target=train, args=(
            rank, args, shared_model, optimizer))
        p.start()
        processes.append(p)
        time.sleep(0.1)
    for p in processes:
        time.sleep(0.1)
        p.join()