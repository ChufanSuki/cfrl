# -*- coding: utf-8 -*-
OMP_NUM_THREADS=1 
import argparse
from multiprocessing import process
import os
import sys
import time
from distutils.util import strtobool
import torch
import torch.optim as optim
import torch.multiprocessing as mp
import torch.nn as nn
import torch.nn.functional as F
from cfrl.wrappers.atari_wrappers import make_atari, wrap_deepmind
from cfrl.nn.atari import AtariLSTMNet, AtariNet
from cfrl.optimizers.shared_rmsprop import SharedRMSprop
from cfrl.optimizers.shared_adam import SharedAdam

from test_a3c import test, train

from torch.utils.tensorboard import SummaryWriter

parser = argparse.ArgumentParser(description='A3C')
parser.add_argument("--exp-name", type=str, default=os.path.basename(__file__).rstrip(".py"),
        help="the name of this experiment")
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
parser.add_argument("--track", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="if toggled, this experiment will be tracked with Weights and Biases")
parser.add_argument("--wandb-project-name", type=str, default="cfrl",
    help="the wandb's project name")
parser.add_argument("--wandb-entity", type=str, default=None,
    help="the entity (team) of wandb's project")

if __name__ == '__main__':
    args = parser.parse_args()
    run_name = f"{args.env_name}__{args.exp_name}__{args.seed}__{int(time.time())}"
    if args.track:
        import wandb

        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=True,
            config=vars(args),
            name=run_name,
            monitor_gym=True,
            save_code=True,
        )
    # writer = SummaryWriter(f"runs/{run_name}")
    # writer.add_text(
    #     "hyperparameters",
    #     "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    # )

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
        elif args.optimizer == 'Adam':
            optimizer = SharedAdam(
                shared_model.parameters(), lr=args.lr)
        else:
            raise ValueError
        optimizer.share_memory()

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