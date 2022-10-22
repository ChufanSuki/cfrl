from cfrl.wrappers.atari_wrappers import make_atari, wrap_deepmind
from cfrl.nn.atari import AtariNet, AtariLSTMNet
import torch
import time
import torch.nn.functional as F
from logging import getLogger
import numpy as np
from cfrl.optimizers.shared_rmsprop import SharedRMSprop
from cfrl.optimizers.shared_adam import SharedAdam

def test(rank, args, shared_model):
    logger = getLogger(__name__)
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
        value, logit = model.forward(obs)
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
            logger.info("Time {0}, episode reward {1}, episode length {2}, reward mean {3}.".format(
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


def train(rank, args, shared_model, optimizer):
    torch.manual_seed(args.seed + rank)
    env = wrap_deepmind(make_atari(args.env_name))
    model = AtariNet(env.observation_space.shape[0], env.action_space)
    observation = env.reset()
    action = env.action_space.sample()
    observation, reward, done, info = env.step(action)
    start_lives = info['ale.lives']

    if optimizer is None:
        if args.optimizer == 'RMSprop':
            optimizer = SharedRMSprop(shared_model.parameters(), lr=args.lr)
        if args.optimizer == 'Adam':
            optimizer = SharedAdam(shared_model.parameters(), lr=args.lr)

    model.train()
    env.seed(args.seed + rank)
    state = env.reset()
    state = torch.from_numpy(np.asarray(state)).float()
    done = True
    lives = start_lives
    episode_length = 0
    while True:
        episode_length += 1
        # Sync with the shared model
        model.load_state_dict(shared_model.state_dict())

        values = []
        log_probs = []
        rewards = []
        entropies = []

        for step in range(args.num_steps):

            value, logit = model(state)
            prob = F.softmax(logit)
            log_prob = F.log_softmax(logit)
            entropy = -(log_prob * prob).sum(1)
            entropies.append(entropy)

            action = prob.multinomial().data
            log_prob = log_prob.gather(1, Variable(action))

            state, reward, done, info = env.step(action.numpy())
            done = done or episode_length >= args.max_episode_length
            if args.count_lives:
                if lives > info['ale.lives']:
                    done = True
            reward = max(min(reward, 1), -1)

            if done:
                episode_length = 0
                lives = start_lives
                state = env.reset()

            state = torch.from_numpy(state).float()
            values.append(value)
            log_probs.append(log_prob)
            rewards.append(reward)

            if done:
                break

        R = torch.zeros(1, 1)
        if not done:

            value, _, _ = model((Variable(state.unsqueeze(0)), (hx, cx)))
            R = value.data

        values.append(Variable(R))
        policy_loss = 0
        value_loss = 0
        R = Variable(R)
        gae = torch.zeros(1, 1)
        for i in reversed(range(len(rewards))):
            R = args.gamma * R + rewards[i]
            advantage = R - values[i]
            value_loss = value_loss + 0.5 * advantage.pow(2)

            # Generalized Advantage Estimataion
            delta_t = rewards[i] + args.gamma * \
                values[i + 1].data - values[i].data
            gae = gae * args.gamma * args.tau + delta_t

            policy_loss = policy_loss - \
                log_probs[i] * Variable(gae) - 0.01 * entropies[i]

        optimizer.zero_grad()

        (policy_loss + 0.5 * value_loss).backward()
        torch.nn.utils.clip_grad_norm(model.parameters(), 40)

        ensure_shared_grads(model, shared_model)
        optimizer.step()
