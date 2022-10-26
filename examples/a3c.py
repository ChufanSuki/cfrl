# -*- coding: utf-8 -*-
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import logging
from logging import getLogger
import wandb
from cfrl.optimizers.shared_adam import SharedAdam
from cfrl.optimizers.shared_rmsprop import SharedRMSprop
from cfrl.nn.atari import AtariLSTMNet, AtariNet
from cfrl.wrappers.atari_wrappers import make_atari, wrap_deepmind
import torch.nn.functional as F
import torch.nn as nn
import torch.multiprocessing as mp
import torch.optim as optim
import torch
from distutils.util import strtobool
import gym
import time
import sys
import os
from multiprocessing import process
import argparse
OMP_NUM_THREADS = 1


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
parser.add_argument('--num-processes', type=int, default=8, metavar='NP',
                    help='how many training processes to use (default: 8)')
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


def test(rank, args, shared_model):
    logger = getLogger(__name__)
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s : %(message)s')
    fileHandler = logging.FileHandler(
        r'{0}{1}_log'.format(args.log_dir, args.env_name), mode='w')
    fileHandler.setFormatter(formatter)
    streamHandler = logging.StreamHandler()
    streamHandler.setFormatter(formatter)
    logger.addHandler(fileHandler)
    logger.addHandler(streamHandler)
    torch.manual_seed(args.seed)
    env = wrap_deepmind(make_atari(args.env_name))
    model = AtariNet(env.observation_space.shape[0], env.action_space)
    model.eval()
    obs = env.reset()
    obs = torch.from_numpy(np.asarray(obs)).float()
    reward_sum = 0
    done = True
    start_time = time.time()
    episode_length = 0
    num_tests = 0
    reward_total_sum = 0
    while True:
        episode_length += 1
        if done:
            model.load_state_dict(shared_model.state_dict())
        value, logit = model(obs)
        prob = F.softmax(logit)
        action = torch.argmax(prob).data.numpy()
        next_obs, r, done, info = env.step(action)
        if episode_length >= args.max_episode_length:
            done = True
        reward_sum += r

        if done:
            num_tests += 1
            reward_total_sum += reward_sum
            reward_mean = reward_total_sum / num_tests
            logger.info("Process {0}: Time {1}, episode reward {2}, episode length {3}, reward mean {4}.".format(
                rank,
                time.strftime("%Hh %Mm %Ss",
                              time.gmtime(time.time() - start_time)),
                reward_sum, episode_length, reward_mean
            ))
            reward_sum = 0
            episode_length = 0
            obs = env.reset()
            time.sleep(60)

        obs = torch.from_numpy(np.asarray(obs)).float()


def ensure_shared_grads(model, shared_model):
    for param, shared_param in zip(model.parameters(), shared_model.parameters()):
        if shared_param.grad is not None:
            return
        shared_param._grad = param.grad


def train(rank, args, shared_model, optimizer, run, run_name):
    # wandb.init(group="A3C")
    logger = getLogger(__name__)
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s : %(message)s')
    fileHandler = logging.FileHandler(
        r'{0}{1}_log'.format(args.log_dir, args.env_name), mode='w')
    fileHandler.setFormatter(formatter)
    streamHandler = logging.StreamHandler()
    streamHandler.setFormatter(formatter)
    logger.addHandler(fileHandler)
    logger.addHandler(streamHandler)
    torch.manual_seed(args.seed + rank)
    env = wrap_deepmind(make_atari(args.env_name))
    env = gym.wrappers.RecordEpisodeStatistics(env)
    env.seed(args.seed + rank)
    env.action_space.seed(args.seed + rank)
    env.observation_space.seed(args.seed + rank)
    if rank == 0:
        env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
    model = AtariNet(env.observation_space.shape[0], env.action_space)
    obs = env.reset()
    action = env.action_space.sample()
    next_obs, reward, done, info = env.step(action)
    start_lives = info['lives']
    assert optimizer is not None
    model.train()
    state = env.reset()
    # state = torch.from_numpy(np.asarray(state)).float()
    done = True
    lives = start_lives
    episode_length = 0
    while True:
        episode_length += 1
        # Sync with the shared model
        model.load_state_dict(shared_model.state_dict())

        actions = []
        values = []
        log_probs = []
        rewards = []
        entropies = []
        episodic_reward = 0

        for step in range(args.num_steps):
            if not torch.is_tensor(state):
                state = torch.from_numpy(np.asarray(state)).float()
            value, logit = model(state)
            prob = F.softmax(logit)
            log_prob = F.log_softmax(logit)
            entropy = -(log_prob * prob).sum()
            entropies.append(entropy)

            action = prob.multinomial(num_samples=1).detach()
            # log_prob = log_prob.gather(1, action)

            state, reward, done, info = env.step(action.data)
            done = done or episode_length >= args.max_episode_length
            if args.count_lives:
                if lives > info['lives']:
                    done = True
            reward = max(min(reward, 1), -1)
            episodic_reward += reward

            values.append(value)
            log_probs.append(log_prob)
            rewards.append(reward)
            actions.append(action)

            if done:
                run.log({"episodic_reward": episodic_reward})
                episode_length = 0
                lives = start_lives
                state = env.reset()
                break

        if done:
            R = 0
        else:
            state = torch.from_numpy(np.asarray(state)).float()
            value, _ = model(state)
            R = value.data

        R = torch.tensor([R])  # shape: (1,)

        values.append(R)
        policy_loss = 0
        value_loss = 0

        # gae = torch.zeros(1, 1)
        for i in reversed(range(len(rewards))):
            R = args.gamma * R + rewards[i]
            advantage = R - values[i]
            value_loss = value_loss + advantage.pow(2)
            policy_loss = policy_loss + \
                log_probs[i][actions[i]] * advantage - 0.01 * entropies[i]
            # Generalized Advantage Estimataion
            # delta_t = rewards[i] + args.gamma * \
            #     values[i + 1].data - values[i].data
            # gae = gae * args.gamma * args.tau + delta_t

            # policy_loss = policy_loss - \
            #     log_probs[i] * gae - 0.01 * entropies[i]

        optimizer.zero_grad()
        loss = policy_loss + 0.5 * value_loss
        loss.backward()
        # torch.nn.utils.clip_grad_norm(model.parameters(), 40)

        ensure_shared_grads(model, shared_model)
        optimizer.step()
        run.log({"loss": loss})


if __name__ == '__main__':
    args = parser.parse_args()
    run_name = f"{args.env_name}__{args.exp_name}__{args.seed}__{int(time.time())}"
    if args.track:
        run = wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=True,
            config=vars(args),
            name=run_name,
            monitor_gym=True,
            save_code=True,
            group="A3C"
        )

    torch.set_default_tensor_type('torch.FloatTensor')
    torch.manual_seed(args.seed)

    env = wrap_deepmind(make_atari(args.env_name))
    shared_model = AtariNet(env.observation_space.shape[0], env.action_space)
    wandb.watch(shared_model)
    if args.load:
        saved_state = torch.load('{0}{1}.dat'.format(
            args.load_model_dir, args.env_name))
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
            rank, args, shared_model, optimizer, run, run_name))
        p.start()
        processes.append(p)
        time.sleep(0.1)
    for p in processes:
        time.sleep(0.1)
        p.join()
