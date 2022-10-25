import argparse
import os
from distutils.util import strtobool
import random
import time

import numpy as np
import torch
import gym
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import torch.optim as optim
from stable_baselines3.common.atari_wrappers import (
    ClipRewardEnv,
    EpisodicLifeEnv,
    FireResetEnv,
    MaxAndSkipEnv,
    NoopResetEnv,
)
from stable_baselines3.common.buffers import ReplayBuffer


def parse_args():
    parser = argparse.ArgumentParser()
    # Experiment settings
    parser.add_argument("--exp-name", type=str, default=os.path.basename(__file__).rstrip(".py"),
                        help="the name of this experiment")
    parser.add_argument("--seed", type=int, default=1,
                        help="seed of the experiment")
    parser.add_argument("--torch-deterministic", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
                        help="if toggled, `torch.backends.cudnn.deterministic=False`")
    parser.add_argument("--cuda", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
                        help="if toggled, cuda will be enabled by default")
    parser.add_argument("--track", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
                        help="if toggled, this experiment will be tracked with Weights and Biases")
    parser.add_argument("--wandb-project-name", type=str, default="cfrl",
                        help="the wandb's project name")
    parser.add_argument("--wandb-entity", type=str, default=None,
                        help="the entity (team) of wandb's project")
    parser.add_argument("--capture-video", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
                        help="whether to capture videos of the agent performances (check out `videos` folder)")
    parser.add_argument("--env-id", type=str, default="BreakoutNoFrameskip-v4",
                        help="the id of the environment")
    # Hyperparameters
    parser.add_argument("--replay-memory-size", type=int, default=1000000,
                        help="SGD updates are sampled from this number of most recent frames")
    parser.add_argument("--agent-history-length", type=int, default=4,
                        help="The number of most recent frames experienced by the agent that are given as input to the Q network")
    parser.add_argument("--action-repeat", type=int, default=4,
                        help="repeate each action selected by the agent this many times")
    parser.add_argument("--minibatch-size", type=int, default=128,
                        help="the batch size of sample from the reply memory")
    parser.add_argument("--update-frequency", type=int, default=4,
                        help="the number of actions selected by the agent between successive SGD updates")
    parser.add_argument("--gamma", type=float, default=0.99,
                        help="the discount factor gamma")
    parser.add_argument("--learning-rate", type=float, default=2.5e-4,
                        help="the learning rate of the optimizer")
    parser.add_argument("--target-network-frequency", type=int, default=10000,
                        help="the timesteps it takes to update the target network")
    parser.add_argument("--initial-exploration", type=float, default=1,
                        help="the starting epsilon for exploration")
    parser.add_argument("--final-exploration", type=float, default=0.1,
                        help="the ending epsilon for exploration")
    parser.add_argument("--final-exploration-frame", type=int, default=1000000,
                        help="the number of frames over which the initial value of epsilon is linearly annealed to its final value")
    parser.add_argument("--replay-start-size", type=int, default=50000,
                        help="A uniform random  policy is run for this number of frames before learning starts and the resulting experience is used to populate the replay memory")
    parser.add_argument("--noop-max", type=int, default=30,
                        help="maximum number of doing nothing action to be performed by the agent at the start of an episode")
    parser.add_argument("--total-timesteps", type=int, default=10000000,
                        help="total timesteps of the experiments")

    args = parser.parse_args()
    # fmt: on
    return args


def make_env(env_id, seed, idx, capture_video, run_name):
    def thunk():
        env = gym.make(env_id)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        if capture_video:
            if idx == 0:
                env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        env = NoopResetEnv(env, noop_max=30)
        env = MaxAndSkipEnv(env, skip=4)
        env = EpisodicLifeEnv(env)
        if "FIRE" in env.unwrapped.get_action_meanings():
            env = FireResetEnv(env)
        env = ClipRewardEnv(env)
        env = gym.wrappers.ResizeObservation(env, (84, 84))
        env = gym.wrappers.GrayScaleObservation(env)
        env = gym.wrappers.FrameStack(env, 4)
        env.seed(seed)
        env.action_space.seed(seed)
        env.observation_space.seed(seed)
        return env

    return thunk


class QNetwork(nn.Module):
    def __init__(self, env):
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv2d(4, 32, 8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(3136, 512),
            nn.ReLU(),
            nn.Linear(512, env.single_action_space.n),
        )

    def forward(self, x):
        return self.network(x/255)


def linear_schedule(start_e: float, end_e: float, duration: int, t: int):
    slope = (end_e - start_e) / duration
    return max(slope * t + start_e, end_e)


if __name__ == "__main__":
    args = parse_args()
    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
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
    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % (
            "\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device(
        "cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    # env setup
    envs = gym.vector.SyncVectorEnv(
        [make_env(args.env_id, args.seed, 0, args.capture_video, run_name)])
    assert isinstance(envs.single_action_space,
                      gym.spaces.Discrete), "only discrete action space is supported"

    q_network = QNetwork(envs).to(device)
    optimizer = optim.RMSprop(q_network.parameters(), lr=args.learning_rate)
    target_network = QNetwork(envs).to(device)
    target_network.load_state_dict(q_network.state_dict())

    rb = ReplayBuffer(
        args.replay_memory_size,
        envs.single_observation_space,
        envs.single_action_space,
        device,
        optimize_memory_usage=True,
        handle_timeout_termination=True,
    )
    start_time = time.time()

    # TRY NOT TO MODIFY: start the game
    obs = envs.reset()
    for global_step in range(args.total_timesteps):
        # ALGO LOGIC: put action logic here
        epsilon = linear_schedule(
            args.initial_exploration, args.final_exploration, args.final_exploration_frame, global_step)
        if random.random() < epsilon:
            actions = np.array([envs.single_action_space.sample()
                               for _ in range(envs.num_envs)])
        else:
            q_values = q_network(torch.Tensor(obs).to(device))
            actions = torch.argmax(q_values, dim=1).cpu().numpy()

        # TRY NOT TO MODIFY: execute the game and log data.
        next_obs, rewards, dones, infos = envs.step(actions)

        # TRY NOT TO MODIFY: record rewards for plotting purposes
        for info in infos:
            if "episode" in info.keys():
                print(
                    f"global_step={global_step}, episodic_return={info['episode']['r']}")
                writer.add_scalar("charts/episodic_return",
                                  info["episode"]["r"], global_step)
                writer.add_scalar("charts/episodic_length",
                                  info["episode"]["l"], global_step)
                writer.add_scalar("charts/epsilon", epsilon, global_step)
                break

        # TRY NOT TO MODIFY: save data to reply buffer; handle `terminal_observation`
        real_next_obs = next_obs.copy()
        for idx, d in enumerate(dones):
            if d:
                real_next_obs[idx] = infos[idx]["terminal_observation"]
        rb.add(obs, real_next_obs, actions, rewards, dones, infos)

        # TRY NOT TO MODIFY: CRUCIAL step easy to overlook
        obs = next_obs

        # ALGO LOGIC: training.
        if global_step > args.replay_start_size and global_step % args.update_frequency == 0:
            data = rb.sample(args.minibatch_size)
            with torch.no_grad():
                target_max, _ = target_network(
                    data.next_observations).max(dim=1)
                td_target = data.rewards.flatten() + args.gamma * target_max * \
                    (1 - data.dones.flatten())
            old_val = q_network(data.observations).gather(
                1, data.actions).squeeze()
            loss = F.mse_loss(td_target, old_val)

            if global_step % 100 == 0:
                writer.add_scalar("losses/td_loss", loss, global_step)
                writer.add_scalar("losses/q_values",
                                  old_val.mean().item(), global_step)
                print("SPS:", int(global_step / (time.time() - start_time)))
                writer.add_scalar(
                    "charts/SPS", int(global_step / (time.time() - start_time)), global_step)

            # optimize the model
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # update the target network
            if global_step % args.target_network_frequency == 0:
                target_network.load_state_dict(q_network.state_dict())

    envs.close()
    writer.close()
